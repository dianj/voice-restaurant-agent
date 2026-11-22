"""Database schema and the synthetic demo data (deterministic: the same seed always gives the same restaurants).

The restaurant table also keeps analytics columns (revenue, costs, review counts) with a few deliberate
gaps and outliers, so the exported CSV can be used for data-cleaning exercises. The agent never reads them.
"""
import csv
import json
import random
import sqlite3
import statistics

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

# Allergen names per language, in the order of ALLERGENS (the official terms used on menus), so the agent
# never translates them itself: in a live test it turned "celery" into the non-word "селера" (целина).
_ALLERGEN_TERMS = {
    "en": ("gluten", "crustaceans", "eggs", "fish", "peanuts", "soy", "milk", "nuts",
           "celery", "mustard", "sesame", "sulphites", "lupin", "molluscs"),
    "bg": ("глутен", "ракообразни", "яйца", "риба", "фъстъци", "соя", "мляко", "ядки",
           "целина", "горчица", "сусам", "сулфити", "лупина", "мекотели"),
    "de": ("Gluten", "Krebstiere", "Eier", "Fisch", "Erdnüsse", "Soja", "Milch", "Schalenfrüchte",
           "Sellerie", "Senf", "Sesam", "Sulfite", "Lupinen", "Weichtiere"),
    "es": ("gluten", "crustáceos", "huevos", "pescado", "cacahuetes", "soja", "leche", "frutos de cáscara",
           "apio", "mostaza", "sésamo", "sulfitos", "altramuces", "moluscos"),
    "fr": ("gluten", "crustacés", "œufs", "poisson", "arachides", "soja", "lait", "fruits à coque",
           "céleri", "moutarde", "sésame", "sulfites", "lupin", "mollusques"),
    "it": ("glutine", "crostacei", "uova", "pesce", "arachidi", "soia", "latte", "frutta a guscio",
           "sedano", "senape", "sesamo", "solfiti", "lupini", "molluschi"),
}
ALLERGEN_NAMES = {lang: dict(zip(ALLERGENS, terms)) for lang, terms in _ALLERGEN_TERMS.items()}
NO_ALLERGENS = {"en": "none declared", "bg": "няма декларирани", "de": "keine angegeben",
                "es": "ninguno declarado", "fr": "aucun déclaré", "it": "nessuno dichiarato"}

# cuisine -> dishes: (names per language, ingredients, allergens, vegetarian, vegan, base price in EUR).
# Dish names are stored in all six languages, so the agent never has to translate a menu itself.
# "en" is also the dish's key in menu_items.dish_name.
def dish(en, bg, de, es, fr, it, ingredients, allergens, vegetarian, vegan, price):
    return ({"en": en, "bg": bg, "de": de, "es": es, "fr": fr, "it": it}, ingredients, allergens, vegetarian, vegan, price)


MENUS = {
    "Italian": [
        dish("Truffle Risotto", "Ризото с трюфел", "Trüffelrisotto", "Risotto con trufa", "Risotto à la truffe",
             "Risotto al tartufo", "arborio rice, black truffle, parmesan, butter, vegetable stock", "milk", True, False, 24.5),
        dish("Margherita Pizza", "Пица Маргарита", "Pizza Margherita", "Pizza margarita", "Pizza margherita",
             "Pizza Margherita", "wheat dough, tomato, mozzarella, basil, olive oil", "gluten, milk", True, False, 13.0),
        dish("Spaghetti Carbonara", "Спагети карбонара", "Spaghetti Carbonara", "Espaguetis a la carbonara",
             "Spaghetti à la carbonara", "Spaghetti alla carbonara",
             "spaghetti, guanciale, egg yolk, pecorino, black pepper", "gluten, eggs, milk", False, False, 16.5),
        dish("Grilled Sea Bass", "Лаврак на скара", "Gegrillter Wolfsbarsch", "Lubina a la parrilla", "Bar grillé",
             "Branzino alla griglia", "sea bass, olive oil, lemon, asparagus, cherry tomatoes", "fish", False, False, 29.0),
        dish("Tiramisu", "Тирамису", "Tiramisu", "Tiramisú", "Tiramisu", "Tiramisù",
             "mascarpone, ladyfingers, espresso, egg, cocoa", "gluten, eggs, milk", True, False, 8.5),
        dish("Minestrone", "Минестроне", "Minestrone", "Minestrone", "Minestrone", "Minestrone",
             "seasonal vegetables, beans, celery, tomato, olive oil", "celery", True, True, 9.0),
    ],
    "Bulgarian": [
        dish("Shopska Salad", "Шопска салата", "Schopska-Salat", "Ensalada shopska", "Salade chopska", "Insalata shopska",
             "tomatoes, cucumbers, roasted peppers, onion, white brined cheese", "milk", True, False, 8.5),
        dish("Kavarma", "Каварма", "Kavarma", "Kavarma", "Kavarma", "Kavarma",
             "pork neck, onions, mushrooms, white wine, savory", "sulphites", False, False, 15.9),
        dish("Tarator", "Таратор", "Tarator", "Tarator", "Tarator", "Tarator",
             "yogurt, cucumber, walnuts, dill, garlic", "milk, nuts", True, False, 5.5),
        dish("Kebapche with Fries", "Кебапчета с пържени картофи", "Kebaptscheta mit Pommes", "Kebapche con patatas fritas",
             "Kebapche et frites", "Kebapche con patatine fritte",
             "minced pork and beef, cumin, potatoes, sunflower oil", "", False, False, 11.0),
        dish("Banitsa", "Баница", "Banitsa", "Banitsa", "Banitsa", "Banitsa",
             "filo pastry, eggs, white brined cheese, yogurt", "gluten, eggs, milk", True, False, 6.0),
        dish("Bean Soup", "Боб чорба", "Bohnensuppe", "Sopa de alubias", "Soupe aux haricots", "Zuppa di fagioli",
             "white beans, carrots, celery, mint, paprika", "celery", True, True, 6.5),
        dish("Boiled Veal with Vegetables", "Телешко варено", "Gekochtes Kalbfleisch mit Gemüse",
             "Ternera hervida con verduras", "Veau bouilli aux légumes", "Vitello bollito con verdure",
             "veal, potatoes, carrots, celery, onion, black peppercorns, parsley", "celery", False, False, 13.5),
        dish("Chicken Soup", "Пилешка супа", "Hühnersuppe", "Sopa de pollo", "Soupe au poulet", "Zuppa di pollo",
             "chicken, carrots, celery, onion, vermicelli, thickened with egg and yogurt",
             "gluten, eggs, milk, celery", False, False, 5.5),
        dish("Moussaka", "Мусака", "Moussaka", "Musaka", "Moussaka", "Moussaka",
             "minced pork and beef, potatoes, onion, tomatoes, topping of yogurt, eggs and flour",
             "gluten, eggs, milk", False, False, 10.5),
        dish("Tripe Soup (Shkembe Chorba)", "Шкембе чорба", "Kuttelsuppe (Schkembe Tschorba)",
             "Sopa de callos (shkembe chorba)", "Soupe de tripes (shkembe chorba)", "Zuppa di trippa (shkembe chorba)",
             "beef tripe, milk, butter, paprika, chili; garlic and vinegar on the side", "milk", False, False, 6.0),
        dish("Sarmi (Stuffed Cabbage Rolls)", "Сарми", "Sarmi (Kohlrouladen)", "Sarmi (rollitos de col rellenos)",
             "Sarmi (feuilles de chou farcies)", "Sarmi (involtini di cavolo)",
             "pickled cabbage leaves, minced pork, rice, onion, paprika, yogurt on the side", "milk", False, False, 11.0),
    ],
    "Japanese": [
        dish("Salmon Nigiri Set", "Нигири със сьомга", "Lachs-Nigiri", "Nigiri de salmón", "Nigiri au saumon",
             "Nigiri di salmone", "salmon, sushi rice, rice vinegar, wasabi", "fish", False, False, 21.0),
        dish("Peanut Ramen", "Рамен с фъстъци", "Erdnuss-Ramen", "Ramen de cacahuete", "Ramen aux cacahuètes",
             "Ramen alle arachidi", "wheat noodles, chicken broth, peanut paste, soft egg, scallions",
             "gluten, peanuts, eggs, soy", False, False, 18.5),
        dish("Vegetable Tempura", "Зеленчукова темпура", "Gemüse-Tempura", "Tempura de verduras", "Tempura de légumes",
             "Tempura di verdure", "seasonal vegetables, wheat batter, soy dipping sauce", "gluten, soy, eggs", True, False, 12.0),
        dish("Edamame", "Едамаме", "Edamame", "Edamame", "Edamame", "Edamame", "soy beans, sea salt", "soy", True, True, 5.0),
        dish("Prawn Gyoza", "Гьоза със скариди", "Garnelen-Gyoza", "Gyozas de langostinos", "Gyoza aux crevettes",
             "Gyoza di gamberi", "prawns, cabbage, wheat wrapper, sesame oil", "crustaceans, gluten, sesame", False, False, 10.5),
        dish("Miso Soup", "Мисо супа", "Misosuppe", "Sopa de miso", "Soupe miso", "Zuppa di miso",
             "miso paste, tofu, wakame, spring onion", "soy", True, True, 4.5),
    ],
    "Mexican": [
        dish("Chicken Tacos", "Тако с пилешко", "Hähnchen-Tacos", "Tacos de pollo", "Tacos au poulet", "Tacos di pollo",
             "corn tortillas, chicken, salsa, coriander, lime", "", False, False, 12.0),
        dish("Beef Burrito", "Бурито с телешко", "Rindfleisch-Burrito", "Burrito de ternera", "Burrito au bœuf",
             "Burrito di manzo", "wheat tortilla, beef, rice, black beans, cheese, sour cream", "gluten, milk", False, False, 14.5),
        dish("Guacamole with Totopos", "Гуакамоле с царевичен чипс", "Guacamole mit Tortilla-Chips", "Guacamole con totopos",
             "Guacamole et chips de maïs", "Guacamole con nachos", "avocado, lime, onion, coriander, corn chips", "",
             True, True, 8.0),
        dish("Vegetable Quesadilla", "Кесадия със зеленчуци", "Gemüse-Quesadilla", "Quesadilla de verduras",
             "Quesadilla aux légumes", "Quesadilla di verdure", "wheat tortilla, peppers, onions, cheddar", "gluten, milk",
             True, False, 11.0),
        dish("Churros", "Чурос", "Churros", "Churros", "Churros", "Churros",
             "wheat dough, sugar, cinnamon, chocolate sauce", "gluten, milk, eggs", True, False, 6.5),
    ],
    "French": [
        dish("French Onion Soup", "Френска лучена супа", "Französische Zwiebelsuppe", "Sopa de cebolla francesa",
             "Soupe à l'oignon", "Zuppa di cipolle alla francese", "onions, beef stock, baguette, gruyere, white wine",
             "gluten, milk, sulphites", False, False, 11.0),
        dish("Coq au Vin", "Кок о вен (петел във вино)", "Coq au Vin", "Coq au vin", "Coq au vin", "Coq au vin",
             "chicken, red wine, mushrooms, bacon, shallots", "sulphites", False, False, 23.0),
        dish("Moules Marinières", "Миди в бяло вино", "Miesmuscheln in Weißwein", "Mejillones a la marinera",
             "Moules marinières", "Cozze alla marinara", "mussels, white wine, garlic, parsley, butter",
             "molluscs, milk, sulphites", False, False, 21.5),
        dish("Ratatouille", "Рататуй", "Ratatouille", "Ratatouille", "Ratatouille", "Ratatouille",
             "aubergine, courgette, peppers, tomato, herbs de Provence", "", True, True, 15.0),
        dish("Crème Brûlée", "Крем брюле", "Crème brûlée", "Crème brûlée", "Crème brûlée", "Crème brûlée",
             "cream, egg yolk, vanilla, caramelised sugar", "milk, eggs", True, False, 8.0),
    ],
    "Indian": [
        dish("Butter Chicken", "Пилешко в масло (Butter Chicken)", "Butter Chicken", "Pollo a la mantequilla",
             "Poulet au beurre", "Pollo al burro", "chicken, tomato, butter, cream, garam masala, rice", "milk", False, False, 16.0),
        dish("Chana Masala", "Чана масала (нахут)", "Chana Masala", "Chana masala", "Chana masala", "Chana masala",
             "chickpeas, tomato, onion, ginger, cumin, rice", "", True, True, 12.5),
        dish("Lamb Rogan Josh", "Агнешко роган джош", "Lamm Rogan Josh", "Cordero rogan josh", "Agneau rogan josh",
             "Agnello rogan josh", "lamb, yogurt, kashmiri chilli, cardamom, rice", "milk", False, False, 19.0),
        dish("Garlic Naan", "Нан с чесън", "Knoblauch-Naan", "Naan de ajo", "Naan à l'ail", "Naan all'aglio",
             "wheat flour, yogurt, garlic, ghee", "gluten, milk", True, False, 3.5),
        dish("Vegetable Samosas", "Самоси със зеленчуци", "Gemüse-Samosas", "Samosas de verduras", "Samoussas aux légumes",
             "Samosa di verdure", "potatoes, peas, wheat pastry, mustard seeds", "gluten, mustard", True, True, 6.0),
    ],
    "Mediterranean": [
        dish("Mezze Platter", "Плато мезе", "Mezze-Platte", "Plato de mezze", "Assiette de mezze", "Piatto di mezze",
             "hummus, baba ganoush, falafel, pita, olives", "gluten, sesame", True, True, 14.0),
        dish("Grilled Octopus", "Октопод на скара", "Gegrillter Oktopus", "Pulpo a la parrilla", "Poulpe grillé",
             "Polpo alla griglia", "octopus, olive oil, lemon, capers, potatoes", "molluscs", False, False, 24.0),
        dish("Greek Salad", "Гръцка салата", "Griechischer Salat", "Ensalada griega", "Salade grecque", "Insalata greca",
             "tomatoes, cucumber, feta, olives, oregano", "milk", True, False, 9.5),
        dish("Lamb Souvlaki", "Агнешко сувлаки", "Lamm-Souvlaki", "Souvlaki de cordero", "Souvlaki d'agneau",
             "Souvlaki di agnello", "lamb, pita, tzatziki, onion, tomato", "gluten, milk", False, False, 17.0),
        dish("Baklava", "Баклава", "Baklava", "Baklava", "Baklava", "Baklava",
             "filo pastry, walnuts, pistachios, honey, butter", "gluten, nuts, milk", True, False, 6.5),
    ],
    "Steakhouse": [
        dish("Ribeye Steak 300 g", "Рибай стек 300 г", "Ribeye-Steak 300 g", "Entrecot ribeye 300 g",
             "Entrecôte ribeye 300 g", "Bistecca ribeye 300 g", "dry-aged beef ribeye, sea salt, rosemary butter",
             "milk", False, False, 34.0),
        dish("Classic Burger", "Класически бургер", "Klassischer Burger", "Hamburguesa clásica", "Burger classique",
             "Hamburger classico", "beef patty, brioche bun, cheddar, pickles, mustard mayo",
             "gluten, milk, eggs, mustard, sesame", False, False, 15.5),
        dish("Caesar Salad", "Салата Цезар", "Caesar Salad", "Ensalada César", "Salade César", "Insalata Caesar",
             "romaine, parmesan, croutons, anchovy dressing, egg", "gluten, milk, eggs, fish", False, False, 11.5),
        dish("Grilled Portobello", "Печурка портобело на скара", "Gegrillter Portobello", "Portobello a la parrilla",
             "Portobello grillé", "Portobello alla griglia", "portobello mushroom, garlic, thyme, olive oil, salad", "",
             True, True, 13.0),
        dish("Cheesecake", "Чийзкейк", "Käsekuchen", "Tarta de queso", "Cheesecake", "Cheesecake",
             "cream cheese, biscuit base, berries", "gluten, milk, eggs", True, False, 8.0),
    ],
}
# How many dishes each restaurant serves from its cuisine's list. Bulgarian kitchens get a fuller menu,
# so the classics (boiled veal, chicken soup, moussaka, sarmi) are found in most of them.
DISHES_PER_RESTAURANT = {"Bulgarian": (7, 10)}
DEFAULT_DISHES_PER_RESTAURANT = (4, 6)

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
    opening_hours TEXT NOT NULL,
    price_per_person INTEGER NOT NULL  -- typical EUR per person, from the menu (see price_per_person())
);
CREATE INDEX idx_restaurants_search ON restaurants (city, cuisine);

CREATE TABLE menu_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    restaurant_id TEXT NOT NULL REFERENCES restaurants (restaurant_id),
    dish_name TEXT NOT NULL,           -- English name
    names TEXT NOT NULL,               -- JSON: dish name per language (en, bg, de, es, fr, it)
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
    extra_names TEXT NOT NULL DEFAULT '[]',
    history TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL CHECK (status IN ('confirmed', 'cancelled')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_reservations_slot ON reservations (restaurant_id, date, status);

CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('event', 'large_group', 'other')),
    customer_name TEXT NOT NULL,
    contact TEXT NOT NULL,
    details TEXT NOT NULL,
    date TEXT,
    guests INTEGER,
    city TEXT,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done')),
    created_at TEXT NOT NULL
);
"""

# Callback requests for a human colleague (events, groups over the online limit, anything the agent cannot do).
REQUESTS_SCHEMA = """CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('event', 'large_group', 'other')),
    customer_name TEXT NOT NULL,
    contact TEXT NOT NULL,
    details TEXT NOT NULL,
    date TEXT,
    guests INTEGER,
    city TEXT,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done')),
    created_at TEXT NOT NULL
);
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
    low, high = DISHES_PER_RESTAURANT.get(restaurant["cuisine"], DEFAULT_DISHES_PER_RESTAURANT)
    picked = rng.sample(dishes, k=min(len(dishes), rng.randint(low, high)))
    return [
        (restaurant["restaurant_id"], names["en"], json.dumps(names, ensure_ascii=False), ingredients, allergens,
         int(veg), int(vegan), round(price * multiplier * rng.uniform(0.9, 1.1), 2))
        for names, ingredients, allergens, veg, vegan, price in picked
    ]


def price_per_person(prices: list) -> int:
    """Typical spend per person in EUR: two courses at the menu's median price, rounded to the nearest 5.

    Computed from the menu (not from average_meal_price, which has deliberate outliers), so it matches
    the prices the guest hears when asking for the menu.
    """
    return max(5, int(5 * round(2 * statistics.median(prices) / 5)))


def build_database(path: str = None, csv_path: str = None) -> dict:
    """(Re)create the database with demo data. Returns row counts."""
    path = path or config.DB_PATH
    rng = random.Random(SEED)
    restaurants = generate_restaurants(rng)
    menu = []
    for r in restaurants:
        items = generate_menu(rng, r)
        r["price_per_person"] = price_per_person([item[-1] for item in items])
        menu += items

    conn = sqlite3.connect(path)
    try:
        with conn:
            conn.executescript("DROP TABLE IF EXISTS requests; DROP TABLE IF EXISTS reservations; "
                               "DROP TABLE IF EXISTS menu_items; DROP TABLE IF EXISTS restaurants;")
            conn.executescript(SCHEMA)
            columns = list(restaurants[0])
            conn.executemany(
                f"INSERT INTO restaurants ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
                [tuple(r[c] for c in columns) for r in restaurants],
            )
            conn.executemany(
                "INSERT INTO menu_items (restaurant_id, dish_name, names, ingredients, allergens, vegetarian, vegan, price) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", menu,
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
    """Create the demo database on first start, or rebuild it when its schema is out of date.

    A rebuild deletes existing reservations; this is demo data, there are no migrations.
    """
    required = {"reservations": "code", "restaurants": "price_per_person", "menu_items": "names"}
    try:
        conn = sqlite3.connect(config.DB_PATH)
        current = all(column in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                      for table, column in required.items())
        conn.close()
    except sqlite3.Error:
        current = False
    if not current:
        build_database()
        return
    # Columns added later are added in place, so existing reservations are kept.
    conn = sqlite3.connect(config.DB_PATH)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(reservations)")}
        for column in ("extra_names", "history"):
            if column not in columns:
                conn.execute(f"ALTER TABLE reservations ADD COLUMN {column} TEXT NOT NULL DEFAULT '[]'")
        conn.executescript(REQUESTS_SCHEMA)
        conn.commit()
    finally:
        conn.close()
