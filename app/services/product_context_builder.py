from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import Settings
from .cache import CachingService, get_caching_service
from .product_gateway_client import ProductGatewayClient


@dataclass
class ProductContextResult:
    context: Dict[str, Any]
    context_hash: str
    cache_hit: bool


class ProductContextBuilder:
    def __init__(
        self,
        *,
        settings: Settings,
        gateway_client: ProductGatewayClient,
        cache: CachingService | None = None,
    ) -> None:
        self._settings = settings
        self._gateway_client = gateway_client
        self._cache = cache or get_caching_service()

    async def get_context(
        self,
        *,
        product_id: str,
        store_id: Optional[str],
        shipping_method: Optional[str],
        authorization: Optional[str],
        trace_id: Optional[str],
        conversation_id: Optional[str],
        user_id: Optional[str],
    ) -> ProductContextResult:
        cached = self._cache.get_product_context(product_id, store_id, shipping_method)
        if cached:
            return ProductContextResult(
                context=cached["context"],
                context_hash=cached["context_hash"],
                cache_hit=True,
            )

        raw = await self._gateway_client.fetch_product_full(
            product_id=product_id,
            authorization=authorization,
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        context = self._build_context(raw)
        context_hash = self._hash_context(context)
        payload = {"context": context, "context_hash": context_hash}
        self._cache.set_product_context(
            product_id,
            store_id,
            shipping_method,
            payload,
            ttl_seconds=self._settings.product_context_ttl_seconds,
        )
        return ProductContextResult(context=context, context_hash=context_hash, cache_hit=False)

    def _build_context(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Build normalized context from GET /api/v1/product-search/{id} response.
        
        Response structure:
        {
            "metadata": {"breadcrumbs": [...]},
            "data": {
                "result": {...product data...},
                "abstractProduct": {...variants...}
            }
        }
        """
        # Extract result from new API structure
        data = raw.get("data") or {}
        result = data.get("result") or raw  # Fallback to raw for backwards compatibility
        metadata = raw.get("metadata") or {}
        abstract_product = data.get("abstractProduct") or {}
        
        additional = result.get("additionalProperties") or {}
        product = {
            "id": result.get("id"),
            "ext_id": result.get("extId"),
            "code": result.get("code"),
            "name": result.get("name"),
            "url": result.get("url"),
            "status": result.get("status"),
            "manufacturer": additional.get("manufacturer"),
            "country": additional.get("country"),
            "prescription": additional.get("prescription"),
            "delivery_available": additional.get("deliveryAvailable"),
            "page_info": self._truncate_text(result.get("pageInfo")),
        }
        product = self._drop_empty(product)

        # Category from breadcrumbs or result.category
        category = self._extract_category(metadata, result)
        if category:
            product["category"] = category

        pricing = {"prices": self._map_prices(result.get("prices") or [])}
        # Add bonuses if present
        if result.get("bonuses"):
            pricing["bonuses"] = result.get("bonuses")
        pricing = self._drop_empty(pricing)

        availability = {
            "stocks": result.get("stocks") or [],
            "in_pickup_store": result.get("inPickupStore"),
            "pickup_stores_count": result.get("pickupStoresCount"),
            "ship_to_store": result.get("shipToStore"),
            "ship_to_stores_count": result.get("shipToStoresCount"),
        }
        availability = self._drop_empty(availability)

        # Delivery/Fulfillment info
        delivery = self._extract_delivery_info(result, additional)
        
        # Return/Warranty policies
        policies = self._extract_policies(result, additional)

        # Process all attributes - this is where the rich data lives
        attributes = self._map_attributes(result.get("attributes") or [])
        
        # Extract key pharmaceutical attributes into separate sections for better LLM access
        pharma_info = self._extract_pharma_info(result.get("attributes") or [])
        
        labels = self._map_labels(result.get("labels") or [])
        documents = self._map_documents(result)
        
        # Product variants from abstractProduct
        variants = self._extract_variants(abstract_product)

        context: Dict[str, Any] = {"product": product}
        if pricing:
            context["pricing"] = pricing
        if availability:
            context["availability"] = availability
        if delivery:
            context["delivery"] = delivery
        if policies:
            context["policies"] = policies
        if pharma_info:
            context["pharma_info"] = pharma_info
        if attributes:
            context["attributes"] = attributes
        if labels:
            context["labels"] = labels
        if documents:
            context["documents"] = documents
        if variants:
            context["variants"] = variants
        return context
    
    def _extract_category(self, metadata: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
        """Extract category from breadcrumbs or result.category."""
        # Try breadcrumbs first
        breadcrumbs = metadata.get("breadcrumbs") or []
        categories = [b.get("name") for b in breadcrumbs if b.get("type") == "CATEGORY"]
        if categories:
            return " > ".join(categories)
        
        # Fallback to result.category
        category = result.get("category") or {}
        return category.get("name")
    
    def _extract_pharma_info(self, attributes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key pharmaceutical attributes for easy LLM access."""
        # Key attribute codes for pharmaceutical products
        pharma_codes = {
            "directions": "indications",           # Показания
            "contraindications": "contraindications",  # Противопоказания
            "sideeffect": "side_effects",          # Побочные действия
            "dosage": "dosage",                    # Способ применения и дозы
            "ingredients": "composition",          # Состав
            "pharmacologiceffect": "pharmacologic_effect",  # Фармакологический эффект
            "pharmocokinetics": "pharmacokinetics",  # Фармакокинетика
            "overdose": "overdose",                # Передозировка
            "interactionwithothers": "drug_interactions",  # Взаимодействие
            "cautions": "special_instructions",    # Особые указания
            "precautionarymeasures": "precautions",  # Меры предосторожности
            "pregnancywarnings": "pregnancy_warnings",  # Беременность
            "storage": "storage",                  # Условия хранения
            "expirationdate": "expiration_months", # Срок годности
            "activeingredient": "active_ingredient",  # Действующее вещество
            "dosageactiveingredient": "active_ingredient_dosage",  # Дозировка ДВ
            "productform": "form",                 # Лекарственная форма
            "quantity": "quantity",                # Количество в упаковке
            "prescription": "prescription_required",  # Рецептурный
        }
        
        pharma_info: Dict[str, Any] = {}
        for attr in attributes:
            code = attr.get("code", "").lower()
            if code in pharma_codes:
                key = pharma_codes[code]
                value = attr.get("value")
                if value:
                    # Truncate long HTML content
                    pharma_info[key] = self._truncate_text(value)
        
        return self._drop_empty(pharma_info)
    
    def _extract_delivery_info(
        self, result: Dict[str, Any], additional: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract delivery/fulfillment information."""
        delivery: Dict[str, Any] = {}
        
        # Basic delivery availability
        if additional.get("deliveryAvailable") is not None:
            delivery["available"] = additional.get("deliveryAvailable")
        
        # Delivery restrictions
        if additional.get("deliveryRestricted"):
            delivery["restricted"] = True
            delivery["restriction_reason"] = additional.get("deliveryRestrictionReason")
        
        # Shipping methods from result
        shipping_methods = result.get("shippingMethods") or result.get("deliveryMethods") or []
        if shipping_methods:
            delivery["methods"] = []
            for method in shipping_methods[:5]:  # Limit to 5 methods
                method_info = {
                    "type": method.get("type") or method.get("code"),
                    "name": method.get("name"),
                    "price": method.get("price"),
                    "min_days": method.get("minDays") or method.get("minDeliveryDays"),
                    "max_days": method.get("maxDays") or method.get("maxDeliveryDays"),
                }
                delivery["methods"].append(self._drop_empty(method_info))
        
        # Express/same-day delivery
        if result.get("expressDelivery") or additional.get("expressDeliveryAvailable"):
            delivery["express_available"] = True
        
        # Pickup options summary
        pickup_count = result.get("pickupStoresCount") or 0
        if pickup_count > 0:
            delivery["pickup_available"] = True
            delivery["pickup_stores_count"] = pickup_count
        
        # Ship to store
        if result.get("shipToStore"):
            delivery["ship_to_store"] = True
            ship_count = result.get("shipToStoresCount")
            if ship_count:
                delivery["ship_to_stores_count"] = ship_count
        
        # Delivery time estimate (if available)
        delivery_time = result.get("deliveryTime") or additional.get("deliveryTimeEstimate")
        if delivery_time:
            delivery["time_estimate"] = delivery_time
        
        return self._drop_empty(delivery)
    
    def _extract_policies(
        self, result: Dict[str, Any], additional: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract return/warranty policies."""
        policies: Dict[str, Any] = {}
        
        # Return policy
        return_policy = result.get("returnPolicy") or additional.get("returnPolicy")
        if return_policy:
            if isinstance(return_policy, dict):
                policies["return"] = {
                    "allowed": return_policy.get("allowed", True),
                    "period_days": return_policy.get("periodDays") or return_policy.get("days"),
                    "conditions": return_policy.get("conditions"),
                }
            else:
                policies["return"] = {"description": str(return_policy)}
        
        # Check attributes for return info
        for attr in result.get("attributes") or []:
            code = (attr.get("code") or "").lower()
            if code in ("returnpolicy", "return_policy", "возврат"):
                policies["return"] = {"description": self._truncate_text(attr.get("value"))}
            elif code in ("warranty", "гарантия", "warrantyperiod"):
                policies["warranty"] = {"description": self._truncate_text(attr.get("value"))}
        
        # Warranty
        warranty = result.get("warranty") or additional.get("warranty")
        if warranty:
            if isinstance(warranty, dict):
                policies["warranty"] = {
                    "period": warranty.get("period") or warranty.get("months"),
                    "type": warranty.get("type"),
                    "description": warranty.get("description"),
                }
            else:
                policies["warranty"] = {"period": str(warranty)}
        
        # Non-returnable flag (common for pharma)
        if additional.get("nonReturnable") or result.get("nonReturnable"):
            policies["return"] = {
                "allowed": False,
                "reason": "Товар не подлежит возврату",
            }
        
        # Prescription drugs typically non-returnable
        if additional.get("prescription"):
            if "return" not in policies:
                policies["return"] = {}
            policies["return"]["note"] = "Рецептурные препараты возврату не подлежат"
        
        return self._drop_empty(policies)
    
    def _extract_variants(self, abstract_product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract product variants info from abstractProduct."""
        if not abstract_product:
            return None
        
        variants_data = abstract_product.get("productsByAttributesV2") or []
        if not variants_data:
            return None
        
        variants: Dict[str, List[str]] = {}
        for variant_group in variants_data:
            name = variant_group.get("name")
            values = variant_group.get("values") or []
            if name and values:
                variant_values = []
                for v in values:
                    val_name = v.get("name")
                    product_info = v.get("product") or {}
                    in_stock = product_info.get("inStock", False)
                    is_current = product_info.get("current", False)
                    if val_name:
                        suffix = ""
                        if is_current:
                            suffix = " (текущий)"
                        elif not in_stock:
                            suffix = " (нет в наличии)"
                        variant_values.append(f"{val_name}{suffix}")
                if variant_values:
                    variants[name] = variant_values
        
        # Add union product info if available
        union_info = abstract_product.get("unionProductInfo") or {}
        if union_info.get("total"):
            variants["_summary"] = {
                "total_variants": union_info.get("total"),
                "price_min": union_info.get("min"),
                "price_max": union_info.get("max"),
            }
        
        return variants if variants else None

    def _map_prices(self, prices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for item in prices:
            payload = {
                "type": item.get("type"),
                "price": item.get("price"),
                "price_no_discount": item.get("priceNoDiscount"),
            }
            payload = self._drop_empty(payload)
            if payload:
                cleaned.append(payload)
        return cleaned

    def _map_attributes(self, attributes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        limit = self._settings.product_context_max_attributes
        cleaned: List[Dict[str, Any]] = []
        for item in attributes[:limit]:
            payload = {
                "code": item.get("code"),
                "name": item.get("name"),
                "value": self._truncate_text(item.get("value")),
            }
            payload = self._drop_empty(payload)
            if payload:
                cleaned.append(payload)
        return cleaned

    def _map_labels(self, labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        limit = self._settings.product_context_max_labels
        cleaned: List[Dict[str, Any]] = []
        for item in labels[:limit]:
            payload = {
                "code": item.get("code"),
                "name": item.get("name"),
                "hint": self._truncate_text(item.get("hint")),
            }
            payload = self._drop_empty(payload)
            if payload:
                cleaned.append(payload)
        return cleaned

    def _map_documents(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        media = raw.get("media") or {}
        instructions = self._map_files(media.get("instructions") or raw.get("instructions") or [])
        certificates = self._map_files(media.get("certificates") or raw.get("certificates") or [])
        documents = {}
        if instructions:
            documents["instructions"] = instructions
        if certificates:
            documents["certificates"] = certificates
        return documents

    def _map_files(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for item in items:
            payload = {
                "name": self._truncate_text(item.get("name") or item.get("title")),
                "url": item.get("url") or item.get("link"),
            }
            payload = self._drop_empty(payload)
            if payload:
                cleaned.append(payload)
        return cleaned

    def _truncate_text(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        max_len = self._settings.product_context_max_text_length
        if len(value) <= max_len:
            return value
        return value[: max_len - 1] + "…"

    @staticmethod
    def _drop_empty(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in payload.items() if value not in (None, "", [], {})}

    @staticmethod
    def _hash_context(context: Dict[str, Any]) -> str:
        serialized = json.dumps(context, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

