from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MockPlatform:
    """File-based mock implementation of the commerce platform."""

    DATA_DIR = Path(__file__).resolve().parent.parent / "mock_data"

    ACTIVE_STATUSES = {"IN_PROGRESS", "READY_FOR_PICKUP"}
    COMPLETED_STATUSES = {"DELIVERED", "PICKED_UP"}

    def __init__(self) -> None:
        self._catalog: List[Dict[str, Any]] = self._load_json("catalog.json", default=[])
        self._carts: Dict[str, Dict[str, Any]] = self._load_json("cart.json", default={})
        self._orders: List[Dict[str, Any]] = self._load_json("orders.json", default=[])
        self._users: List[Dict[str, Any]] = self._load_json("users.json", default=[])
        self._addresses: List[Dict[str, Any]] = self._load_json("addresses.json", default=[])
        self._pharmacies: List[Dict[str, Any]] = self._load_json("pharmacies.json", default=[])
        self._favorites: Dict[str, List[str]] = self._load_json("favorites.json", default={})

        self._product_index: Dict[str, Dict[str, Any]] = {
            product["id"]: product for product in self._catalog if product.get("id")
        }
        self._pharmacy_index: Dict[str, Dict[str, Any]] = {
            pharmacy["id"]: pharmacy for pharmacy in self._pharmacies if pharmacy.get("id")
        }
        self._user_index: Dict[str, Dict[str, Any]] = {
            user["id"]: user for user in self._users if user.get("id")
        }

    # -------------------------------------------------------------------------
    # Generic helpers
    # -------------------------------------------------------------------------
    def _load_json(self, filename: str, *, default: Any) -> Any:
        path = self.DATA_DIR / filename
        if not path.exists():
            logger.info("Mock data file %s is missing, using defaults.", filename)
            return deepcopy(default)
        try:
            with path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to decode %s: %s", filename, exc)
            return deepcopy(default)

    def _write_json(self, filename: str, data: Any) -> None:
        path = self.DATA_DIR / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

    def _product_snapshot(self, product_id: str) -> Dict[str, Any] | None:
        product = self._product_index.get(product_id)
        if not product:
            return None
        return deepcopy(product)

    def _resolve_cart_key(self, user_id: Optional[str]) -> str:
        if user_id and user_id in self._carts:
            return user_id
        return "default"

    def _ensure_cart_entry(self, key: str) -> Dict[str, Any]:
        cart = self._carts.setdefault(
            key,
            {
                "cart_id": f"cart-{key}",
                "currency": "RUB",
                "items": [],
            },
        )
        return cart

    def _cart_payload(self, raw_cart: Dict[str, Any]) -> Dict[str, Any]:
        payload = deepcopy(raw_cart)
        items = payload.setdefault("items", [])
        total = 0.0
        for item in items:
            product_info = self._product_index.get(item.get("product_id", ""))
            if product_info:
                item.setdefault("title", product_info.get("name"))
                item.setdefault("price", product_info.get("price", 0.0))
                item.setdefault("image_url", product_info.get("image_url"))
            qty = item.get("qty") or 0
            price = item.get("price") or 0.0
            total += qty * price
        payload["total"] = round(total, 2)
        payload.setdefault("currency", "RUB")
        return payload

    def _order_timestamp(self, order: Dict[str, Any]) -> Optional[datetime]:
        for key in ("completed_at", "created_at"):
            value = order.get(key)
            if not value:
                continue
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                continue
        return None

    # -------------------------------------------------------------------------
    # Catalog helpers
    # -------------------------------------------------------------------------
    def list_products(self) -> List[Dict[str, Any]]:
        return [deepcopy(product) for product in self._catalog]

    def get_product(self, product_id: str | None) -> Dict[str, Any] | None:
        if not product_id:
            return None
        return self._product_snapshot(product_id)

    # -------------------------------------------------------------------------
    # User profile
    # -------------------------------------------------------------------------
    def get_user_profile(self, user_id: str) -> Dict[str, Any] | None:
        user = self._user_index.get(user_id)
        if not user:
            return None
        return deepcopy(user)

    def get_user_addresses(self, user_id: str) -> List[Dict[str, Any]]:
        return [deepcopy(addr) for addr in self._addresses if addr.get("user_id") == user_id]

    def get_pharmacy(self, pharmacy_id: str) -> Dict[str, Any] | None:
        pharmacy = self._pharmacy_index.get(pharmacy_id)
        if not pharmacy:
            return None
        return deepcopy(pharmacy)

    def list_pharmacies(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = [deepcopy(pharmacy) for pharmacy in self._pharmacies]
        if limit is None:
            return items
        return items[:limit]

    # -------------------------------------------------------------------------
    # Favorites
    # -------------------------------------------------------------------------
    def get_favorites(self, user_id: str) -> List[Dict[str, Any]]:
        product_ids = self._favorites.get(user_id, [])
        favorites: List[Dict[str, Any]] = []
        for product_id in product_ids:
            product = self._product_snapshot(product_id)
            if product:
                favorites.append(product)
        return favorites

    def add_favorite(self, user_id: str, product_id: str) -> List[Dict[str, Any]]:
        if not self._product_snapshot(product_id):
            logger.info("Cannot add favorite: product %s not found", product_id)
            return self.get_favorites(user_id)
        if user_id not in self._favorites:
            self._favorites[user_id] = []
        if product_id not in self._favorites[user_id]:
            self._favorites[user_id].append(product_id)
            self._write_json("favorites.json", self._favorites)
        return self.get_favorites(user_id)

    def remove_favorite(self, user_id: str, product_id: str) -> List[Dict[str, Any]]:
        items = self._favorites.get(user_id)
        if items and product_id in items:
            items.remove(product_id)
            self._write_json("favorites.json", self._favorites)
        return self.get_favorites(user_id)

    # -------------------------------------------------------------------------
    # Cart
    # -------------------------------------------------------------------------
    def get_cart(self, user_id: Optional[str]) -> Dict[str, Any]:
        key = self._resolve_cart_key(user_id)
        cart = self._ensure_cart_entry(key)
        return self._cart_payload(cart)

    def add_to_cart(self, user_id: Optional[str], product_id: str, qty: int = 1) -> Dict[str, Any]:
        if not product_id:
            return self.get_cart(user_id)
        if not self.get_product(product_id):
            return self.get_cart(user_id)
        key = self._resolve_cart_key(user_id)
        cart = self._ensure_cart_entry(key)
        items = cart.setdefault("items", [])
        line = next((item for item in items if item.get("product_id") == product_id), None)
        qty = max(qty, 1)
        if line:
            line["qty"] = int(line.get("qty", 0)) + qty
        else:
            product_snapshot = self._product_snapshot(product_id)
            price = product_snapshot.get("price", 0.0) if product_snapshot else 0.0
            items.append(
                {
                    "product_id": product_id,
                    "qty": qty,
                    "price": price,
                    "title": product_snapshot.get("name") if product_snapshot else None,
                    "image_url": product_snapshot.get("image_url") if product_snapshot else None,
                }
            )
        self._write_json("cart.json", self._carts)
        return self._cart_payload(cart)

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------
    def get_orders(self, user_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        bucket = []
        status_filter: set[str] | None = None
        normalized = status.lower() if status else None
        if normalized == "active":
            status_filter = self.ACTIVE_STATUSES
        elif normalized == "completed":
            status_filter = self.COMPLETED_STATUSES
        elif normalized == "cancelled":
            status_filter = {"CANCELLED"}
        for order in self._orders:
            if order.get("user_id") != user_id:
                continue
            if status_filter and order.get("status") not in status_filter:
                continue
            bucket.append(deepcopy(order))
        bucket.sort(key=lambda entry: self._order_timestamp(entry) or datetime.min, reverse=True)
        return bucket

    def get_order_by_id(self, order_id: str) -> Dict[str, Any] | None:
        for order in self._orders:
            if order.get("order_id") == order_id:
                return deepcopy(order)
        return None

    def get_recent_purchases(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for order in self._orders:
            if order.get("user_id") != user_id:
                continue
            if order.get("status") not in self.COMPLETED_STATUSES:
                continue
            order_ts = self._order_timestamp(order)
            for item in order.get("items") or []:
                product_id = item.get("product_id")
                if not product_id:
                    continue
                bucket = stats.setdefault(
                    product_id,
                    {
                        "count": 0,
                        "last_purchase_at": order_ts,
                        "product": self._product_snapshot(product_id),
                        "fallback": deepcopy(item),
                    },
                )
                bucket["count"] += int(item.get("qty") or 0) or 1
                if order_ts and (
                    bucket["last_purchase_at"] is None or order_ts > bucket["last_purchase_at"]
                ):
                    bucket["last_purchase_at"] = order_ts
        if not stats:
            return []
        sorted_items = sorted(
            stats.values(),
            key=lambda entry: (
                -entry["count"],
                -(entry["last_purchase_at"].timestamp() if entry["last_purchase_at"] else 0),
            ),
        )
        result: List[Dict[str, Any]] = []
        for entry in sorted_items[:limit]:
            product = entry["product"] or {}
            if not product:
                product = {
                    "id": entry["fallback"].get("product_id"),
                    "name": entry["fallback"].get("title"),
                }
            product["purchase_count"] = entry["count"]
            if entry["last_purchase_at"]:
                product["last_purchase_at"] = entry["last_purchase_at"].isoformat()
            result.append(product)
        return result

    # -------------------------------------------------------------------------
    # Product detailed info
    # -------------------------------------------------------------------------
    def get_product_instructions(self, product_id: str) -> str | None:
        product = self._product_index.get(product_id)
        if not product:
            return None
        return product.get("instructions")

    def get_product_contraindications(self, product_id: str) -> str | None:
        product = self._product_index.get(product_id)
        if not product:
            return None
        return product.get("contraindications")

    def get_product_specifications(self, product_id: str) -> Dict[str, Any] | None:
        product = self._product_index.get(product_id)
        if not product:
            return None
        specs = product.get("specifications", {})
        return {
            "manufacturer": product.get("manufacturer"),
            "country": product.get("country"),
            "dosage_form": product.get("dosage_form"),
            "storage": product.get("storage"),
            "expiration": product.get("expiration"),
            "side_effects": product.get("side_effects"),
            **specs,
        }

    def get_product_availability_by_pharmacies(
        self, product_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get availability of a product across multiple pharmacies."""
        product = self._product_index.get(product_id)
        if not product:
            return []
        
        result: List[Dict[str, Any]] = []
        for pharmacy in self._pharmacies[:limit]:
            product_pharmacy = product.get("pharmacy_id")
            is_available = product_pharmacy is None or product_pharmacy == pharmacy.get("id")
            availability = {
                "pharmacy_id": pharmacy.get("id"),
                "pharmacy_name": pharmacy.get("name"),
                "address": pharmacy.get("address"),
                "metro": pharmacy.get("metro"),
                "working_hours": pharmacy.get("working_hours"),
                "in_stock": is_available and product.get("availability") == "in_stock",
                "stock_qty": product.get("stock_qty", 0) if is_available else 0,
                "price": product.get("price"),
            }
            result.append(availability)
        return result

    def find_products_by_inn(self, inn: str) -> List[Dict[str, Any]]:
        """Find products by INN (International Nonproprietary Name)."""
        if not inn:
            return []
        inn_lower = inn.lower()
        return [
            deepcopy(product)
            for product in self._catalog
            if product.get("inn") and inn_lower in product.get("inn", "").lower()
        ]

    def find_analogs(self, product_id: str) -> List[Dict[str, Any]]:
        """Find analog products for a given product."""
        product = self._product_index.get(product_id)
        if not product:
            return []
        
        analogs: List[Dict[str, Any]] = []
        analog_ids = product.get("analogs_ids", [])
        for analog_id in analog_ids:
            analog = self._product_snapshot(analog_id)
            if analog:
                analogs.append(analog)
        
        product_inn = product.get("inn")
        if product_inn:
            for other in self._catalog:
                if other.get("id") == product_id:
                    continue
                if other.get("id") in analog_ids:
                    continue
                if other.get("inn") and other.get("inn").lower() == product_inn.lower():
                    analogs.append(deepcopy(other))
        
        return analogs

    def find_promo_products(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Find products with active promotions."""
        promo_products = [
            deepcopy(product)
            for product in self._catalog
            if product.get("promo_flags")
        ]
        promo_products.sort(key=lambda p: p.get("score", 0), reverse=True)
        return promo_products[:limit]

    # -------------------------------------------------------------------------
    # Pharmacies extended
    # -------------------------------------------------------------------------
    def find_pharmacies_by_metro(self, metro_station: str) -> List[Dict[str, Any]]:
        """Find pharmacies near a metro station."""
        if not metro_station:
            return []
        metro_lower = metro_station.lower()
        return [
            deepcopy(pharmacy)
            for pharmacy in self._pharmacies
            if pharmacy.get("metro") and metro_lower in pharmacy.get("metro", "").lower()
        ]

    def get_pharmacy_working_hours(self, pharmacy_id: str) -> str | None:
        pharmacy = self._pharmacy_index.get(pharmacy_id)
        if not pharmacy:
            return None
        return pharmacy.get("working_hours")

    def find_nearest_pharmacy_with_product(
        self, product_id: str, user_region: str | None = None
    ) -> Dict[str, Any] | None:
        """Find nearest pharmacy that has the product in stock."""
        product = self._product_index.get(product_id)
        if not product:
            return None
        
        candidates = self._pharmacies
        if user_region:
            candidates = [p for p in candidates if p.get("region_id") == user_region] or candidates
        
        for pharmacy in candidates:
            product_pharmacy = product.get("pharmacy_id")
            if product_pharmacy is None or product_pharmacy == pharmacy.get("id"):
                if product.get("availability") == "in_stock":
                    return {
                        "pharmacy": deepcopy(pharmacy),
                        "product": deepcopy(product),
                        "stock_qty": product.get("stock_qty", 0),
                    }
        return None

    # -------------------------------------------------------------------------
    # Orders extended
    # -------------------------------------------------------------------------
    def create_order(
        self, user_id: str, items: List[Dict[str, Any]], delivery_type: str = "pickup"
    ) -> Dict[str, Any]:
        """Create a new order from cart items."""
        import uuid
        
        order_id = f"ord-{uuid.uuid4().hex[:8]}"
        total = sum((item.get("price", 0) * item.get("qty", 1)) for item in items)
        
        order = {
            "order_id": order_id,
            "user_id": user_id,
            "items": deepcopy(items),
            "total": round(total, 2),
            "currency": "RUB",
            "status": "IN_PROGRESS",
            "delivery_type": delivery_type,
            "created_at": datetime.now().isoformat(),
            "estimated_delivery": "1-2 рабочих дня" if delivery_type == "delivery" else "Сегодня",
        }
        
        self._orders.append(order)
        self._write_json("orders.json", self._orders)
        return order

    def cancel_order(self, order_id: str, reason: str | None = None) -> Dict[str, Any] | None:
        """Cancel an order."""
        for order in self._orders:
            if order.get("order_id") == order_id:
                if order.get("status") in self.COMPLETED_STATUSES:
                    return None
                order["status"] = "CANCELLED"
                order["cancelled_at"] = datetime.now().isoformat()
                order["cancellation_reason"] = reason or "По запросу пользователя"
                self._write_json("orders.json", self._orders)
                return deepcopy(order)
        return None

    def extend_order_pickup_time(self, order_id: str) -> Dict[str, Any] | None:
        """Extend pickup time for an order."""
        for order in self._orders:
            if order.get("order_id") == order_id:
                if order.get("status") != "READY_FOR_PICKUP":
                    return None
                order["pickup_extended"] = True
                order["pickup_deadline"] = datetime.now().replace(hour=23, minute=59).isoformat()
                self._write_json("orders.json", self._orders)
                return deepcopy(order)
        return None

    def reorder(self, user_id: str, order_id: str) -> Dict[str, Any] | None:
        """Create a new order based on a previous order."""
        original = self.get_order_by_id(order_id)
        if not original or original.get("user_id") != user_id:
            return None
        items = original.get("items", [])
        if not items:
            return None
        return self.create_order(user_id=user_id, items=items, delivery_type=original.get("delivery_type", "pickup"))

    # -------------------------------------------------------------------------
    # Cart extended
    # -------------------------------------------------------------------------
    def clear_cart(self, user_id: Optional[str]) -> Dict[str, Any]:
        """Clear all items from cart."""
        key = self._resolve_cart_key(user_id)
        cart = self._ensure_cart_entry(key)
        cart["items"] = []
        self._write_json("cart.json", self._carts)
        return self._cart_payload(cart)

    def remove_from_cart(self, user_id: Optional[str], product_id: str) -> Dict[str, Any]:
        """Remove a product from cart."""
        key = self._resolve_cart_key(user_id)
        cart = self._ensure_cart_entry(key)
        cart["items"] = [item for item in cart.get("items", []) if item.get("product_id") != product_id]
        self._write_json("cart.json", self._carts)
        return self._cart_payload(cart)

    def apply_promo_code(self, user_id: Optional[str], promo_code: str) -> Dict[str, Any]:
        """Apply a promo code to cart."""
        key = self._resolve_cart_key(user_id)
        cart = self._ensure_cart_entry(key)
        
        promo_codes = {
            "WELCOME10": {"discount_percent": 10, "description": "Скидка 10% на первый заказ"},
            "SUMMER20": {"discount_percent": 20, "description": "Летняя скидка 20%"},
            "FREE_DELIVERY": {"free_delivery": True, "description": "Бесплатная доставка"},
        }
        
        promo = promo_codes.get(promo_code.upper())
        if not promo:
            return {**self._cart_payload(cart), "promo_error": "Промокод не найден или истёк"}
        
        cart["applied_promo"] = {"code": promo_code.upper(), **promo}
        self._write_json("cart.json", self._carts)
        
        payload = self._cart_payload(cart)
        if promo.get("discount_percent"):
            discount = payload["total"] * promo["discount_percent"] / 100
            payload["discount"] = round(discount, 2)
            payload["total_with_discount"] = round(payload["total"] - discount, 2)
        return payload

    # -------------------------------------------------------------------------
    # User extended
    # -------------------------------------------------------------------------
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any] | None:
        """Update user preferences."""
        user = self._user_index.get(user_id)
        if not user:
            return None
        user_prefs = user.setdefault("preferences", {})
        user_prefs.update(preferences)
        self._write_json("users.json", self._users)
        return deepcopy(user)

    def get_user_coupons(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active coupons for a user."""
        return [
            {"code": "BIRTHDAY15", "description": "Скидка 15% ко дню рождения", "discount_percent": 15, "valid_until": "2025-12-31", "min_order": 500},
            {"code": "LOYAL5", "description": "Скидка 5% для постоянных клиентов", "discount_percent": 5, "valid_until": "2025-06-30", "min_order": 0},
        ]

    # -------------------------------------------------------------------------
    # Booking
    # -------------------------------------------------------------------------
    def book_product_pickup(self, user_id: str, product_id: str, pharmacy_id: str) -> Dict[str, Any] | None:
        """Book a product for pickup at a specific pharmacy."""
        import uuid
        product = self._product_index.get(product_id)
        pharmacy = self._pharmacy_index.get(pharmacy_id)
        
        if not product or not pharmacy:
            return None
        if product.get("availability") != "in_stock":
            return None
        
        booking_id = f"book-{uuid.uuid4().hex[:8]}"
        return {
            "booking_id": booking_id,
            "product": deepcopy(product),
            "pharmacy": deepcopy(pharmacy),
            "status": "RESERVED",
            "reserved_until": datetime.now().replace(hour=21, minute=0).isoformat(),
            "message": f"Товар забронирован до 21:00 сегодня в аптеке {pharmacy.get('name')}",
        }
