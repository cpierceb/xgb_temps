import subprocess
import os

# cities = [
#     ("amsterdam", "netherlands"),
#     ("basel", "switzerland"),
#     ("berlin", "germany"),
#     ("bern", "switzerland"),
#     ("biel", "switzerland"),
#     ("birmingham", "uk"),
#     ("freiburg", "germany"),
#     ("ghent", "belgium"),
#     ("novisad", "serbia"),
#     ("rennes", "france"),
#     ("turku", "finland"),
#     ("zurich", "switzerland")
# ]

cities = [
    ("bern", "switzerland"),
    ("biel", "switzerland"),
    ("lausanne", "switzerland"),
    ("thun", "switzerland"),
    ("winterthur", "switzerland"),
    ("zurich", "switzerland")
]


from multiprocessing import Pool

def run_city(city_country):
    city, country = city_country
    env = os.environ.copy()
    env['PIPELINE_TARGET_CITY'] = city
    env['PIPELINE_TARGET_COUNTRY'] = country
    try:
        result = subprocess.run(['python', 'xgb_a_main.py'], env=env, check=True, text=True)
        return (city, True, None)
    except subprocess.CalledProcessError as e:
        return (city, False, e.stderr)

# replace the for loop with:
with Pool(processes=len(cities)) as pool:
    results = pool.map(run_city, cities)

    
# for city, country in cities:
#     print(f"\n{'='*60}")
#     print(f"Processing: {city.upper()}, {country.upper()}")
#     print(f"{'='*60}\n")
    
#     # Set environment variables
#     env = os.environ.copy()
#     env['PIPELINE_TARGET_CITY'] = city
#     env['PIPELINE_TARGET_COUNTRY'] = country
    
#     try:
#         # Run the script
#         result = subprocess.run(
#             ['python', 'xgb_a_main.py'],
#             env=env,
#             check=True,
#             # capture_output=True,
#             text=True
#         )
#         print(result.stdout)
#         print(f"✓ Successfully processed {city}")
        
#     except subprocess.CalledProcessError as e:
#         print(f"✗ Error processing {city}:")
#         print(e.stderr)
#         # Continue to next city even if this one fails
#         continue

print("\n" + "="*60)
print("All cities processed!")
print("="*60)