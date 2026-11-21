import sqlite3
import random
import pandas as pd

def generate_data():
    cuisines = ["Italian", "Bulgarian", "Japanese", "Mexican", "French", "Indian", "Mediterranean", "Steakhouse"]
    locations = ["Sofia Center", "Lozenets", "Vitosha Blvd", "Plovdiv Old Town", "Varna Sea Garden", "London Soho", "Berlin Mitte"]
    prefixes = ["Golden", "Urban", "Royal", "Bistro", "La", "The Secret", "Grand", "Artisan", "Blue", "Velvet"]
    suffixes = ["Table", "Kitchen", "Garden", "House", "Grill", "Terrace", "Corner", "Lounge", "Plate", "Oak"]
    price_ranges = ["$", "$$", "$$$", "$$$$"]

    restaurants = []
    for i in range(1, 1001):
        name = f"{random.choice(prefixes)} {random.choice(suffixes)} #{i}"
        cuisine = random.choice(cuisines)
        price_range = random.choice(price_ranges)
        
        rating = None if random.random() < 0.04 else round(random.uniform(3.2, 4.9), 1)
        review_count = random.randint(15, 1200) if random.random() > 0.02 else 18500
        avg_meal = round(random.uniform(15, 95), 2) if random.random() > 0.01 else 450.00
        capacity = random.choice([20, 35, 50, 75, 100, 120, 150])
        delivery = random.choice([True, False])
        opening_year = None if random.random() < 0.03 else random.randint(1998, 2025)
        monthly_rev = round(capacity * avg_meal * random.uniform(12, 25), 2)
        op_costs = round(monthly_rev * random.uniform(0.55, 0.85), 2)
        hours = random.choice(["10:00 - 22:00", "11:30 - 23:30", "08:00 - 21:00", "12:00 - 00:00"])

        restaurants.append({
            "restaurant_id": f"REST-{i:04d}",
            "name": name,
            "location": random.choice(locations),
            "cuisine": cuisine,
            "price_range": price_range,
            "rating": rating,
            "review_count": review_count,
            "average_meal_price": avg_meal,
            "seating_capacity": capacity,
            "delivery_available": delivery,
            "opening_year": opening_year,
            "monthly_revenue": monthly_rev,
            "operating_costs": op_costs,
            "opening_hours": hours
        })

    df = pd.DataFrame(restaurants)
    df.to_csv("restaurants_1000.csv", index=False)

    conn = sqlite3.connect("restaurant_agent.db")
    df.to_sql("restaurants", conn, if_exists="replace", index=False)

    menu_items = [
        ("REST-0001", "Truffle Risotto", "Arborio rice, black truffle, parmesan, butter, vegetable stock", "Dairy", 24.50),
        ("REST-0001", "Grilled Sea Bass", "Fresh sea bass, olive oil, lemon, asparagus, cherry tomatoes", "Fish", 32.00),
        ("REST-0001", "Gluten-Free Quinoa Salad", "Quinoa, avocado, roasted peppers, pumpkin seeds, lemon dressing", "None", 16.00),
        ("REST-0002", "Shopska Salad", "Tomatoes, cucumbers, roasted peppers, onion, white brined cheese", "Dairy", 11.50),
        ("REST-0002", "Kavarma Kebap", "Pork neck, onions, mushrooms, white wine, savory spices", "Sulfites", 18.90),
        ("REST-0003", "Salmon Nigiri Set", "Atlantic salmon, sushi rice, rice vinegar, wasabi", "Fish", 22.00),
        ("REST-0003", "Peanut Ramen", "Wheat noodles, chicken broth, peanut paste, soft egg, scallions", "Gluten, Peanuts, Eggs", 19.50)
    ]
    conn.execute("DROP TABLE IF EXISTS menu_items")
    conn.execute("""
        CREATE TABLE menu_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            restaurant_id TEXT,
            dish_name TEXT,
            ingredients TEXT,
            allergens TEXT,
            price REAL
        )
    """)
    conn.executemany("INSERT INTO menu_items (restaurant_id, dish_name, ingredients, allergens, price) VALUES (?, ?, ?, ?, ?)", menu_items)

    conn.execute("DROP TABLE IF EXISTS reservations")
    conn.execute("""
        CREATE TABLE reservations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            restaurant_id TEXT,
            customer_name TEXT,
            date TEXT,
            time TEXT,
            guests INTEGER,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("Готово! Създадени са restaurants_1000.csv и restaurant_agent.db")

if __name__ == "__main__":
    generate_data()
