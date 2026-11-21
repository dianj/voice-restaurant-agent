"""Database schema and the synthetic demo data (deterministic: the same seed always gives the same restaurants).

The restaurant table also keeps analytics columns (revenue, costs, review counts) with a few deliberate
gaps and outliers, so the exported CSV can be used for data-cleaning exercises. The agent never reads them.
"""
import csv
import random
import sqlite3

from . import config

SEED = 42
RESTAURANT_COUNT = 1000

# city -> areas
LOCATIONS = {
    "Sofia": ["Center", "Lozenets", "Vitosha Blvd"],
    "Plovdiv": ["Old Town", "Kapana"],
    "Varna": ["Sea Garden", "Center"],
    "London": ["Soho"],
    "Berlin": ["Mitte"],
}
CUISINES = ["Italian", "Bulgarian", "Japanese", "Mexican", "French", "Indian", "Mediterranean", "Steakhouse"]
PRICE_RANGES = {"$": 0.7, "$$": 1.0, "$$$": 1.4, "$$$$": 2.0}  # price multiplier for menu items
OPENING_HOURS = ["10:00-22:00", "11:30-23:30", "08:00-21:00", "12:00-00:00", "17:00-23:00"]
CAPACITIES = [20, 35, 50, 75, 100, 120]

NAME_FIRST = ["Golden", "Urban", "Royal", "Blue", "Velvet", "Silver", "Olive", "Copper", "Rustic", "Little",
              "Green", "Old", "Wild", "Sunny", "Hidden", "Crimson", "Amber", "Cedar", "Iron", "Linen",
              "Saffron", "Juniper", "Marble", "Ivory", "Black", "White", "Red", "Lucky", "Quiet", "Grand"]
NAME_SECOND = ["Table", "Kitchen", "Garden", "House", "Grill", "Terrace", "Corner", "Lounge", "Plate", "Oak",
               "Spoon", "Fork", "Lantern", "Courtyard", "Pantry", "Hearth", "Vine", "Harbor", "Mill", "Bakery",
               "Tavern", "Bistro", "Cellar", "Orchard", "Fig", "Pepper", "Salt", "Ember", "Willow", "Anchor",
               "Barrel", "Clove", "Basil", "Thyme"]

# The 14 allergens that EU law requires restaurants to declare.
ALLERGENS = ("gluten", "crustaceans", "eggs", "fish", "peanuts", "soy", "milk", "nuts",
             "celery", "mustard", "sesame", "sulphites", "lupin", "molluscs")

# cuisine -> (dish, ingredients, allergens, vegetarian, vegan, base price in EUR)
MENUS = {
    "Italian": [
        ("Truffle Risotto", "arborio rice, black truffle, parmesan, butter, vegetable stock", "milk", True, False, 24.5),
        ("Margherita Pizza", "wheat dough, tomato, mozzarella, basil, olive oil", "gluten, milk", True, False, 13.0),
        ("Spaghetti Carbonara", "spaghetti, guanciale, egg yolk, pecorino, black pepper", "gluten, eggs, milk", False, False, 16.5),
        ("Grilled Sea Bass", "sea bass, olive oil, lemon, asparagus, cherry tomatoes", "fish", False, False, 29.0),
        ("Tiramisu", "mascarpone, ladyfingers, espresso, egg, cocoa", "gluten, eggs, milk", True, False, 8.5),
        ("Minestrone", "seasonal vegetables, beans, celery, tomato, olive oil", "celery", True, True, 9.0),
    ],
    "Bulgarian": [
        ("Shopska Salad", "tomatoes, cucumbers, roasted peppers, onion, white brined cheese", "milk", True, False, 8.5),
        ("Kavarma", "pork neck, onions, mushrooms, white wine, savory", "sulphites", False, False, 15.9),
        ("Tarator", "yogurt, cucumber, walnuts, dill, garlic", "milk, nuts", True, False, 5.5),
        ("Kebapche with Fries", "minced pork and beef, cumin, potatoes, sunflower oil", "", False, False, 11.0),
        ("Banitsa", "filo pastry, eggs, white brined cheese, yogurt", "gluten, eggs, milk", True, False, 6.0),
        ("Bean Soup", "white beans, carrots, celery, mint, paprika", "celery", True, True, 6.5),
    ],
    "Japanese": [
        ("Salmon Nigiri Set", "salmon, sushi rice, rice vinegar, wasabi", "fish", False, False, 21.0),
        ("Peanut Ramen", "wheat noodles, chicken broth, peanut paste, soft egg, scallions", "gluten, peanuts, eggs, soy", False, False, 18.5),
        ("Vegetable Tempura", "seasonal vegetables, wheat batter, soy dipping sauce", "gluten, soy, eggs", True, False, 12.0),
        ("Edamame", "soy beans, sea salt", "soy", True, True, 5.0),
        ("Prawn Gyoza", "prawns, cabbage, wheat wrapper, sesame oil", "crustaceans, gluten, sesame", False, False, 10.5),
        ("Miso Soup", "miso paste, tofu, wakame, spring onion", "soy", True, True, 4.5),
    ],
    "Mexican": [
        ("Chicken Tacos", "corn tortillas, chicken, salsa, coriander, lime", "", False, False, 12.0),
        ("Beef Burrito", "wheat tortilla, beef, rice, black beans, cheese, sour cream", "gluten, milk", False, False, 14.5),
        ("Guacamole with Totopos", "avocado, lime, onion, coriander, corn chips", "", True, True, 8.0),
        ("Vegetable Quesadilla", "wheat tortilla, peppers, onions, cheddar", "gluten, milk", True, False, 11.0),
        ("Churros", "wheat dough, sugar, cinnamon, chocolate sauce", "gluten, milk, eggs", True, False, 6.5),
    ],
    "French": [
        ("French Onion Soup", "onions, beef stock, baguette, gruyere, white wine", "gluten, milk, sulphites", False, False, 11.0),
        ("Coq au Vin", "chicken, red wine, mushrooms, bacon, shallots", "sulphites", False, False, 23.0),
        ("Moules Marinieres", "mussels, white wine, garlic, parsley, butter", "molluscs, milk, sulphites", False, False, 21.5),
        ("Ratatouille", "aubergine, courgette, peppers, tomato, herbs de Provence", "", True, True, 15.0),
        ("Creme Brulee", "cream, egg yolk, vanilla, caramelised sugar", "milk, eggs", True, False, 8.0),
    ],
    "Indian": [
        ("Butter Chicken", "chicken, tomato, butter, cream, garam masala, rice", "milk", False, False, 16.0),
        ("Chana Masala", "chickpeas, tomato, onion, ginger, cumin, rice", "", True, True, 12.5),
        ("Lamb Rogan Josh", "lamb, yogurt, kashmiri chilli, cardamom, rice", "milk", False, False, 19.0),
        ("Garlic Naan", "wheat flour, yogurt, garlic, ghee", "gluten, milk", True, False, 3.5),
        ("Vegetable Samosas", "potatoes, peas, wheat pastry, mustard seeds", "gluten, mustard", True, True, 6.0),
    ],
    "Mediterranean": [
        ("Mezze Platter", "hummus, baba ganoush, falafel, pita, olives", "gluten, sesame", True, True, 14.0),
        ("Grilled Octopus", "octopus, olive oil, lemon, capers, potatoes", "molluscs", False, False, 24.0),
        ("Greek Salad", "tomatoes, cucumber, feta, olives, oregano", "milk", True, False, 9.5),
        ("Lamb Souvlaki", "lamb, pita, tzatziki, onion, tomato", "gluten, milk", False, False, 17.0),
        ("Baklava", "filo pastry, walnuts, pistachios, honey, butter", "gluten, nuts, milk", True, False, 6.5),
    ],
    "Steakhouse": [
        ("Ribeye Steak 300 g", "dry-aged beef ribeye, sea salt, rosemary butter", "milk", False, False, 34.0),
        ("Classic Burger", "beef patty, brioche bun, cheddar, pickles, mustard mayo", "gluten, milk, eggs, mustard, sesame", False, False, 15.5),
        ("Caesar Salad", "romaine, parmesan, croutons, anchovy dressing, egg", "gluten, milk, eggs, fish", False, False, 11.5),
        ("Grilled Portobello", "portobello mushroom, garlic, thyme, olive oil, salad", "", True, True, 13.0),
        ("Cheesecake", "cream cheese, biscuit base, berries", "gluten, milk, eggs", True, False, 8.0),
    ],
}

SCHEMA = """
CREATE TABLE restaurants (
    restaurant_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    area TEXT NOT NULL,
    cuisine TEXT NOT NULL,
    price_range TEXT NOT NULL,
    rating REAL,
    review_count INTEGER,
    average_meal_price REAL,
    seating_capacity INTEGER NOT NULL CHECK (seating_capacity > 0),
    delivery_available INTEGER NOT NULL,
    opening_year INTEGER,
    monthly_revenue REAL,
    operating_costs REAL,
    opening_hours TEXT NOT NULL
);
CREATE INDEX idx_restaurants_search ON restaurants (city, cuisine);

CREATE TABLE menu_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    restaurant_id TEXT NOT NULL REFERENCES restaurants (restaurant_id),
    dish_name TEXT NOT NULL,
    ingredients TEXT NOT NULL,
    allergens TEXT NOT NULL,
    vegetarian INTEGER NOT NULL,
    vegan INTEGER NOT NULL,
    price REAL NOT NULL
);
CREATE INDEX idx_menu_restaurant ON menu_items (restaurant_id);

CREATE TABLE reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    restaurant_id TEXT NOT NULL REFERENCES restaurants (restaurant_id),
    customer_name TEXT NOT NULL,
    phone TEXT,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    guests INTEGER NOT NULL CHECK (guests > 0),
    notes TEXT,
    status TEXT NOT NULL CHECK (status IN ('confirmed', 'cancelled')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_reservations_slot ON reservations (restaurant_id, date, status);
"""


def generate_restaurants(rng: random.Random) -> list:
    names = [f"{a} {b}" for a in NAME_FIRST for b in NAME_SECOND]
    rng.shuffle(names)
    places = [(city, area) for city, areas in LOCATIONS.items() for area in areas]
    rows = []
    for i in range(1, RESTAURANT_COUNT + 1):
        city, area = rng.choice(places)
        capacity = rng.choice(CAPACITIES)
        avg_meal = round(rng.uniform(15, 95), 2) if rng.random() > 0.01 else 450.00   # outlier
        monthly_revenue = round(capacity * avg_meal * rng.uniform(12, 25), 2)
        rows.append({
            "restaurant_id": f"REST-{i:04d}",
            "name": names[i - 1],
            "city": city,
            "area": area,
            "cuisine": rng.choice(CUISINES),
            "price_range": rng.choice(list(PRICE_RANGES)),
            "rating": None if rng.random() < 0.04 else round(rng.uniform(3.2, 4.9), 1),  # new places: no rating yet
            "review_count": rng.randint(15, 1200) if rng.random() > 0.02 else 18500,     # outlier
            "average_meal_price": avg_meal,
            "seating_capacity": capacity,
            "delivery_available": int(rng.random() < 0.5),
            "opening_year": None if rng.random() < 0.03 else rng.randint(1998, 2025),
            "monthly_revenue": monthly_revenue,
            "operating_costs": round(monthly_revenue * rng.uniform(0.55, 0.85), 2),
            "opening_hours": rng.choice(OPENING_HOURS),
        })
    return rows


def generate_menu(rng: random.Random, restaurant: dict) -> list:
    dishes = MENUS[restaurant["cuisine"]]
    multiplier = PRICE_RANGES[restaurant["price_range"]]
    picked = rng.sample(dishes, k=min(len(dishes), rng.randint(4, 6)))
    return [
        (restaurant["restaurant_id"], name, ingredients, allergens, int(veg), int(vegan),
         round(price * multiplier * rng.uniform(0.9, 1.1), 2))
        for name, ingredients, allergens, veg, vegan, price in picked
    ]


def build_database(path: str = None, csv_path: str = None) -> dict:
    """(Re)create the database with demo data. Returns row counts."""
    path = path or config.DB_PATH
    rng = random.Random(SEED)
    restaurants = generate_restaurants(rng)
    menu = [item for r in restaurants for item in generate_menu(rng, r)]

    conn = sqlite3.connect(path)
    try:
        with conn:
            conn.executescript("DROP TABLE IF EXISTS reservations; DROP TABLE IF EXISTS menu_items; DROP TABLE IF EXISTS restaurants;")
            conn.executescript(SCHEMA)
            columns = list(restaurants[0])
            conn.executemany(
                f"INSERT INTO restaurants ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
                [tuple(r[c] for c in columns) for r in restaurants],
            )
            conn.executemany(
                "INSERT INTO menu_items (restaurant_id, dish_name, ingredients, allergens, vegetarian, vegan, price) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", menu,
            )
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("restaurants", "menu_items", "reservations")}
    finally:
        conn.close()

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(restaurants[0]))
            writer.writeheader()
            writer.writerows(restaurants)
    return counts


def ensure_database():
    """Create the demo database on first start, or if it is from the old prototype schema."""
    try:
        conn = sqlite3.connect(config.DB_PATH)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(reservations)")}
        conn.close()
    except sqlite3.Error:
        columns = set()
    if "code" not in columns:
        build_database()
