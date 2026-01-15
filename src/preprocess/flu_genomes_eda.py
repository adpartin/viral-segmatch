"""
Exploratory Data Analysis (EDA) for Flu genomes metadata.

Inputs
------
This script works with two related tab-delimited files in `data/raw/Full_Flu_Annos/`:

1) `Flu_Genomes.key`
   - `hash_id`: internal genome ID used to link records
   - `virus_name`: free-text name, usually containing `A/<host?>/<location>/<strain>/<year>`
   - `hn_subtype`: e.g. H1N1, H3N2, H5N1
   - `seg_1` ... `seg_8`: BV-BRC genome segment IDs (often two IDs per segment)

2) `Flu.first-seg.meta.tab`
   - `hash_id`, `virus_name`, `hn_subtype`, `first_seg_id`
   - `host_common_name`: natural host where virus was isolated (preferred over inferred host)
   - `lab_host`, `passage`: sparse propagation/passaging metadata

Core parsing outputs
--------------------
We parse `virus_name` (from `Flu_Genomes.key`) into coarse metadata:
- `host_inferred`, `year`, `strain`
- `geo_location_inferred`: *one* extracted location token from the name

We then standardize that location token into:
- `geo_location_raw`: the extracted token before standardization (same as `geo_location_inferred`)
- `geo_location_canonical`: the standardized canonical token produced by `standardize_location()`
- `geo_city`, `geo_state`, `geo_country`: best-effort components inferred from the canonical token
- `geo_location_reject_reason`: why standardization returned no canonical token (debugging aid)

Finally we produce `geo_location_clean` for downstream filtering, driven by a configurable policy:
- default policy `auto_us_state_else_country`: if `geo_country == "USA"`, use `geo_state` when present;
  otherwise use `geo_country` when present; fallback to the canonical token.

Important limitations / ambiguity
---------------------------------
- Only ONE location token is extracted per `virus_name`. We cannot reliably infer “county” unless the
  token itself is a county-like string and we have an explicit mapping.
- Some names are inherently ambiguous (e.g., "Rochester", "Georgia"). We only disambiguate when we have
  explicit mappings (e.g., "Republic of Georgia") or unambiguous abbreviations (e.g., GA vs the country name).

Outputs
-------
Outputs are saved under `data/processed/flu/metadata_eda/`:
- `flu_genomes_metadata_parsed.csv`: full merged dataset with parsed + standardized columns
- `*_count.csv`: value-count distributions (host/subtype/location/passage)
- `plots/`: summary plots (host/subtype/year/location distributions; temporal trends; host×subtype heatmap)
"""

import sys
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d
import importlib

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


# ============================================================================
# MODULE-LEVEL CONSTANTS FOR GEO-LOCATION AND HOST PARSING
# ============================================================================

# Base animal/bird terms shared between host_keywords and NON_LOCATION_TERMS
BASE_ANIMAL_BIRD_TERMS = [
    # Birds
    'chicken', 'duck', 'mallard', 'turkey', 'goose', 'quail', 'pheasant', 'guinea',
    'turnstone', 'sandpiper', 'gull', 'swan', 'eagle', 'hawk', 'falcon', 'tern',
    'scaup', 'teal', 'scoter', 'pintail', 'grackle', 'blackbird', 'robin', 'sparrow',
    'pigeon', 'dove', 'pelican', 'loon', 'vulture', 'crow', 'peafowl', 'francolin',
    'grouse', 'ptarmigan', 'woodcock', 'snipe', 'plover', 'oystercatcher', 'avocet',
    'stilt', 'godwit', 'curlew', 'willet', 'dowitcher', 'phalarope', 'jaeger',
    'shearwater', 'petrel', 'albatross', 'gannet', 'cormorant', 'anhinga', 'booby',
    'frigatebird', 'tropicbird', 'gadwall', 'sanderling', 'shoveler', 'bufflehead',
    'widgeon', 'wigeon', 'redhead', 'canvasback', 'merganser', 'pochard', 'eider',
    'garganey', 'whimbrel', 'dunlin', 'knot', 'raven', 'magpie', 'ibis', 'egret',
    'heron', 'stork', 'flamingo', 'grebe', 'coot', 'rail', 'gallinule', 'auk',
    'puffin', 'murre', 'guillemot', 'razorbill', 'cuckoo', 'roadrunner', 'nightjar',
    'swift', 'hummingbird', 'kingfisher', 'woodpecker', 'flycatcher', 'shrike',
    'vireo', 'jay', 'nutcracker', 'chickadee', 'titmouse', 'bushtit', 'wren',
    'gnatcatcher', 'kinglet', 'bluebird', 'thrush', 'catbird', 'mockingbird',
    'thrasher', 'starling', 'myna', 'waxwing', 'phainopepla', 'warbler', 'tanager',
    'cardinal', 'grosbeak', 'bunting', 'junco', 'towhee', 'longspur', 'snowflake',
    'finch', 'crossbill', 'redpoll', 'siskin', 'goldfinch', 'partridge', 'turtledove',
    'ostrich', 'penguin', 'shorebird', 'great horned owl',
    # Mammals and other animals
    'ferret', 'seal', 'whale', 'horse', 'equine', 'canine', 'dog', 'cat', 'feline',
    'skunk', 'raccoon', 'fox', 'lion', 'tiger', 'dolphin', 'alpaca', 'rhea', 'emu',
    'dairy cow', 'dairy', 'cow', 'cattle', 'bovine', 'goat', 'sheep', 'ram',
    'serval', 'mink', 'pika', 'bear', 'swine', 'pig',
]

# Additional host-specific terms (not in NON_LOCATION_TERMS)
HOST_SPECIFIC_TERMS = [
    'human',  # Recognized as host but not filtered as non-location
    'poultry',  # General category for domesticated birds
]

# Host keywords = base terms + host-specific terms
HOST_KEYWORDS = BASE_ANIMAL_BIRD_TERMS + HOST_SPECIFIC_TERMS

# Scientific names (genus species format) - used in NON_LOCATION_TERMS
SCIENTIFIC_NAMES = [
    'gallus gallus', 'gallus gallus domesticus', 'gallus', 'meleagris gallopavo',
    'anser fabalis', 'anser albifrons', 'anser brachyrhynchus', 'anser cygnoides',
    'cygnus olor', 'cygnus columbianus', 'cygnus cygnus', 'cygnus atratus',
    'equus caballus', 'equus ferus caballus', 'mustela putorius furo', 'neovison vison',
    'canis lupus familiaris', 'sus scrofa', 'bos taurus', 'alopochen aegyptiacus',
    'pelecanus thagus', 'pelecanus', 'thalasseus acuflavidus', 'thalasseus maximus',
    'larus argentatus', 'larus michahellis', 'chroicocephalus ridibundus',
    'corvus ossifragus', 'falco peregrinus', 'falco tinnunculus', 'falco_rusticolus',
    'buteo buteo', 'accipiter gentilis schvedowi', 'buteogallus urubitinga',
    'himantopus himantopus', 'pluvialis dominica', 'calidris_fuscicollis',
    'calidris ruficollis', 'spatula querquedula', 'chlidonias hybrida',
    'bucefala clangula', 'aix galericulata', 'numida meleagris', 'pavo cristatus',
    'anthropoides virgo', 'syhrrhaptes paradoxus', 'copsychus saularis',
    'fregata magnificens', 'rousettus aegyptiacus', 'rattus norvegicus',
    'panthera leo', 'gallinula chloropus', 'athene noctua', 'homo sapiens',
    'homo sapien', 'camelus dromedarius',
]

# Environmental/non-geographic terms
ENVIRONMENTAL_TERMS = [
    'water', 'air', 'env', 'enviroment', 'feces', 'surface water', 'bioaerosol',
    'pet food', 'raw pet food', 'environment',
]

# Additional non-location terms (not in base animal/bird terms)
ADDITIONAL_NON_LOCATION_TERMS = [
    # General bird/animal categories
    'aquatic bird', 'backyard bird', 'wild bird', 'wild waterbird', 'wild waterfowl',
    'wildbird', 'wild-bird', 'domestic bird', 'migratory bird', 'seabird', 'bird',
    'avian', 'broiler', 'bovine_milk', 'donkey', 'camel', 'meerkat', 'cheetah',
    'sloth bear', 'stone marten', 'muskrat', 'weasel', 'ermine', 'norway rat',
    # Specific bird species (common names)
    'common eiders', 'common eider', 'common pochard', 'common porchard',
    'common goldeneye', 'common murre', 'common raven', 'common buzzard',
    'common magpie', 'common kestrel', 'common merganser', 'common grebe',
    'crane', 'great-horned owl', 'owl', 'little egret', 'northern shoveler',
    'northern shoverl', 'red knot', 'red shoveler', 'red-breasted merganser',
    'ruddy turnstone', 'scaly-breasted munia', 'red-necked stint',
    'red-necked grebe', 'rosy-billed pochard', 'red crested pochard',
    'little grebe', 'horned grebe', 'eared grebe', 'great crested grebe',
    'great crested-grebe', 'great grebe', 'great grabe', 'eurasian coot',
    'red-gartered coot', 'white-faced ibis', 'great egret', 'snowy egret',
    'grey heron', 'grey-faced buzzard', 'green heron', 'night heron',
    'purple heron', 'striated heron', 'black skimmer', 'black brant',
    'black-billed magpie', 'black-legged kittiwake', 'black swift', 'blue jay',
    'barn swallow', 'swallow', 'european starling', 'common raven', 'rook',
    'pied magpie', 'chukar', 'chukkar', 'chukka', 'peacock', 'chilean flamingo',
    'condor', 'california condor', 'king eider', 'steller\'s eider',
    'spectacled eider', 'hooded merganser', 'brant', 'cape shoveler',
    'gentoo penguin', 'adelie penguin', 'african penguin', 'humboldt penguin',
    'thick-billed murre', 'thick-billed_murre', 'open-billed stork',
    'open-bill stork', 'openbill stork', 'white stork', 'oriental white stork',
    'marabou stork', 'american wood stork', 'american coot', 'american widgeon',
    'american_wigeon', 'dark-eyed junco', 'ferruginous pochard',
    'comon goldeneye', 'great tit', 'great bustard', 'great grabe',
    'ground jay', 'long-billed calandra', 'long-tailed shrike',
    'crested caracara', 'chimango caracara', 'roseate spoonbill', 'watercock',
    'brambling', 'twite', 'shrike', 'buzzard', 'raptor', 'fighting bird',
    'softbill', 'woodpeckers', 'chinese hwamei', 'white-backed munia',
    'white bellied bustard', 'houbara bustard', 'tibetan snowfinch',
    'black-faced bunting', 'japanese white-eye', 'yellow-headed amazon',
    'parrot', 'piegon', 'chiken', 'backyard poultry',
    # Additional bird species
    'laughing gull', 'ring-necked duck', 'ring necked duck', 'american wigeon',
    'eurasian wigeon', 'snowy owl', 'great blue heron', 'sacred ibis',
    'buff-necked ibis', 'black-crowned night heron', 'chinstrap penguin',
    'chinstrap_penguin', 'crested goshawk', 'crested myna',
    # Other animals/hosts
    'plateau pika', 'polar bear', 'black bear', 'mouse', 'giant panda',
    'giant anteater', 'owston\'s civet', 'dark fruit-eating bat',
    'little yellow-shouldered bat', 'flat-faced bat', 'burmeister\'s porpoise',
    'fisher', 'cougar', 'lynx', 'bat',
    # Host/human terms (for filtering, not parsing)
    'homo sapiens', 'homo sapien',
    # Parsing errors / invalid terms
    'old layer', 'reassortant', 'reverse genetics',
]

# NON_LOCATION_TERMS = base terms + scientific names + environmental + additional
NON_LOCATION_TERMS = sorted(list(set(
    BASE_ANIMAL_BIRD_TERMS + SCIENTIFIC_NAMES + ENVIRONMENTAL_TERMS + ADDITIONAL_NON_LOCATION_TERMS
)))

# Fast lookup sets used during parsing
NON_LOCATION_TERMS_SET = set(t.lower() for t in NON_LOCATION_TERMS)
ENVIRONMENTAL_TERMS_SET = set(t.lower() for t in ENVIRONMENTAL_TERMS + ['reassortant'])
SCIENTIFIC_GENUS_SET = set(
    (n.split()[0].split("_")[0].lower())
    for n in SCIENTIFIC_NAMES
    if isinstance(n, str) and n.strip()
)

# Bird/animal endings for compound name detection (e.g., "Great Horned Owl", "laughing gull")
BIRD_ANIMAL_ENDINGS = [
    'gull', 'duck', 'goose', 'swan', 'owl', 'hawk', 'eagle', 'falcon',
    'heron', 'egret', 'ibis', 'stork', 'pelican', 'cormorant', 'tern',
    'grebe', 'loon', 'coot', 'rail', 'plover', 'sandpiper', 'turnstone',
    'knot', 'dunlin', 'sanderling', 'merganser', 'shoveler', 'wigeon', 'widgeon',
    'teal', 'pintail', 'gadwall', 'scaup', 'goldeneye', 'bufflehead',
    'eider', 'scoter', 'pochard', 'raven', 'magpie', 'crow', 'jay',
    'flamingo', 'penguin', 'auk', 'puffin', 'murre', 'guillemot',
    'razorbill', 'dove', 'pigeon', 'cuckoo', 'swift', 'hummingbird',
    'kingfisher', 'woodpecker', 'flycatcher', 'shrike', 'vireo', 'thrush',
    'robin', 'starling', 'warbler', 'tanager', 'cardinal', 'grosbeak',
    'bunting', 'sparrow', 'finch', 'partridge', 'turtledove', 'whimbrel',
    'garganey', 'redhead', 'canvasback', 'godwit', 'curlew', 'willet',
    'avocet', 'stilt', 'oystercatcher', 'phalarope', 'dowitcher', 'jaeger',
    'shearwater', 'petrel', 'albatross', 'gannet', 'booby', 'frigatebird',
    'tropicbird',
    'mouse',
]

# US State abbreviations mapping (single source of truth)
US_STATE_ABBREV = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
}

# Derived from US_STATE_ABBREV
VALID_US_STATE_ABBREVIATIONS = sorted(list(US_STATE_ABBREV.keys()))
US_STATES_LOWERCASE = [v.lower() for v in US_STATE_ABBREV.values()]

# US territories (appear in this dataset as 2-letter abbreviations like states)
US_TERRITORY_ABBREV = {
    "PR": "Puerto Rico",
    "GU": "Guam",
    "VI": "U.S. Virgin Islands",
    "AS": "American Samoa",
    "MP": "Northern Mariana Islands",
}
VALID_US_TERRITORY_ABBREVIATIONS = sorted(list(US_TERRITORY_ABBREV.keys()))

# Common multi-word locations
COMMON_MULTIWORD_LOCATIONS = [
    'new york city', 'north carolina', 'south carolina', 'new jersey',
    'new hampshire', 'new mexico', 'west virginia', 'north dakota',
    'south dakota', 'rhode island', 'district of columbia',
    'viet nam', 'hong kong', 'new zealand', 'south korea', 'south africa'
]

# Common locations (subset of US states + multi-word)
COMMON_LOCATIONS = [
    'new york', 'rhode island', 'north carolina', 'south carolina',
    'new jersey', 'new hampshire', 'new mexico', 'west virginia',
    'north dakota', 'south dakota', 'washington', 'california',
    'massachusetts', 'pennsylvania', 'district of columbia'
]

# City to State/Country mapping
CITY_MAPPING = {
    # US Cities -> States
    'Chicago': 'Illinois',
    'New York City': 'New York',
    'Houston': 'Texas',
    'Memphis': 'Tennessee',
    'Baltimore': 'Maryland',
    'Boston': 'Massachusetts',
    'Los Angeles': 'California',
    'San Diego': 'California',
    'Santa Clara': 'California',
    'King County': 'Washington',
    'Snohomish County': 'Washington',
    'Rockville Illinois': 'Illinois',  # Handle "Rockville Illinois" -> Illinois
    'Rockville': 'Illinois',  # Rockville -> Illinois
    # International Cities -> Countries
    'Lyon': 'France',
    'London': 'United Kingdom',
    'Sydney': 'Australia',
    'Brisbane': 'Australia',
    'Perth': 'Australia',
    'Auckland': 'New Zealand',
    'Wellington': 'New Zealand',
    'Bangkok': 'Thailand',
    'Helsinki': 'Finland',
    'Moscow': 'Russia',
    'Santiago': 'Chile',
    'Mexico City': 'Mexico',
    'Taipei': 'Taiwan',
    'Kaohsiung': 'Taiwan',
    'Shanghai': 'China',
    'Beijing': 'China',
    'HaNoi': 'Vietnam',
    'Newcastle': 'Australia',
    # Additional major cities
    'Kyoto': 'Japan',
    'Tokyo': 'Japan',
    'Osaka': 'Japan',
    'Madrid': 'Spain',
    'Barcelona': 'Spain',
    'Paris': 'France',
    'Rome': 'Italy',
    'Milan': 'Italy',
    'Berlin': 'Germany',
    'Munich': 'Germany',
    'Amsterdam': 'Netherlands',
    'Stockholm': 'Sweden',
    'Copenhagen': 'Denmark',
    'Seoul': 'South Korea',
    'Manila': 'Philippines',
    'Jakarta': 'Indonesia',
    'Buenos Aires': 'Argentina',
    'Sao Paulo': 'Brazil',
    'Rio de Janeiro': 'Brazil',
    'Lima': 'Peru',
    'Bogota': 'Colombia',
    'Cairo': 'Egypt',
    'Nairobi': 'Kenya',
    'Lagos': 'Nigeria',
    'Johannesburg': 'South Africa',
    'Dubai': 'United Arab Emirates',
    'Tel Aviv': 'Israel',
    'Ankara': 'Turkey',
    'Istanbul': 'Turkey',
    'Athens': 'Greece',
    'Warsaw': 'Poland',
    'Prague': 'Czech Republic',
    'Vienna': 'Austria',
    'Budapest': 'Hungary',
    'Bucharest': 'Romania',
    'Kuala Lumpur': 'Malaysia',
    'Singapore': 'Singapore',
    'Dhaka': 'Bangladesh',
    'Karachi': 'Pakistan',
    'Mumbai': 'India',
    'Delhi': 'India',
    'Hanoi': 'Vietnam',
    'Ho Chi Minh City': 'Vietnam',
    'Managua': 'Nicaragua',
    'Porto Alegre': 'Brazil',
    'Canberra': 'Australia',
    'Parma': 'Italy',
    'Kamigoto': 'Japan',
    'Aomori': 'Japan',
    'Linkou': 'Taiwan',
    'Wenzhou': 'China',
    'Huizhou': 'China',
    'Nanchang': 'China',
    'Qingdao': 'China',
    'Wuhan': 'China',
    'Guangzhou': 'China',
    'Nanjing': 'China',
    'Jeddah': 'Saudi Arabia',
    'Lohne': 'Germany',
    'Pyongyang': 'North Korea',
}

# Province/State abbreviation to full name mapping
PROVINCE_ABBREV = {
    'ALB': 'Alberta',  # Alberta, Canada
}

# Province/State to Country mapping
PROVINCE_TO_COUNTRY = {
    # Canadian Provinces
    'Alberta': 'Canada',
    'Manitoba': 'Canada',
    'Ontario': 'Canada',
    'Quebec': 'Canada',
    'Saskatchewan': 'Canada',
    'Nova Scotia': 'Canada',
    'New Brunswick': 'Canada',
    'British Columbia': 'Canada',
    # Australian States
    'Victoria': 'Australia',
    'Queensland': 'Australia',
    'South Australia': 'Australia',
    'Western Australia': 'Australia',
    'New South Wales': 'Australia',
    'Tasmania': 'Australia',
    'Northern Territory': 'Australia',
    'Australian Capital Territory': 'Australia',
    'Canterbury': 'New Zealand',
    'Waikato': 'New Zealand',
    # Chinese Provinces (first-level administrative divisions, similar to US states)
    # Keep this list broad; otherwise provinces leak into geo_country-missing bucket.
    'Anhui': 'China',
    'Beijing': 'China',
    'Chongqing': 'China',
    'Fujian': 'China',
    'Gansu': 'China',
    'Guangdong': 'China',
    'Guangxi': 'China',
    'Guizhou': 'China',
    'Hainan': 'China',
    'Hebei': 'China',
    'Heilongjiang': 'China',
    'Henan': 'China',
    'Hubei': 'China',
    'Hunan': 'China',
    'Inner Mongolia': 'China',
    'Jiangsu': 'China',
    'Jiangxi': 'China',
    'Jilin': 'China',
    'Liaoning': 'China',
    'Ningxia': 'China',
    'Qinghai': 'China',
    'Shaanxi': 'China',
    'Shandong': 'China',
    'Shanghai': 'China',
    'Shanxi': 'China',
    'Sichuan': 'China',
    'Tianjin': 'China',
    'Tibet': 'China',
    'Xinjiang': 'China',
    'Yunnan': 'China',
    'Zhejiang': 'China',
    # Common city-level strings that show up as if they were provinces
    'Dongguan': 'China',
    'Shantou': 'China',
    'Shenzhen': 'China',
    'Eastern China': 'China',
    # Taiwan admin areas often appear as the sole location token
    'Yunlin': 'Taiwan',
    'Tainan': 'Taiwan',
    # Japan prefectures / regions
    'Hokkaido': 'Japan',
    # Germany federal states (full names)
    'Rheinland-Pfalz': 'Germany',
    'Bavaria': 'Germany',
    'Hamburg': 'Germany',
    'Rostov-on-Don': 'Russia',
    # Mexico states (can appear without the "Mexico-" prefix)
    'Jalisco': 'Mexico',
    # US territories / special cases
    'Puerto Rico': 'USA',
    'Guam': 'USA',
    # Japan prefectures (common in this dataset)
    'Niigata': 'Japan',
    'Hiroshima': 'Japan',
    'Aichi': 'Japan',
    'Hyogo': 'Japan',
    'Miyazaki': 'Japan',
    # Russia regions/cities (at least country should be correct even if admin level is fuzzy)
    'Krasnodar': 'Russia',
    'Altai': 'Russia',
    # German States (map Germany-XX to Germany)
}

# Country-level locations (no state/province, just country)
COUNTRY_ONLY_LOCATIONS = {
    'USA': 'USA',
    'Netherlands': 'Netherlands',
    'Vietnam': 'Vietnam',
    'Italy': 'Italy',
    'Singapore': 'Singapore',
    'Bangladesh': 'Bangladesh',
    'Nicaragua': 'Nicaragua',
    'China': 'China',
    'Peru': 'Peru',
    'Sweden': 'Sweden',
    'France': 'France',
    'Egypt': 'Egypt',
    'Denmark': 'Denmark',
    'Taiwan': 'Taiwan',
    'Thailand': 'Thailand',
    'South Korea': 'South Korea',
    'Korea': 'South Korea',
    'Mexico': 'Mexico',
    'Chile': 'Chile',
    'England': 'United Kingdom',
    # Disambiguation: make the country explicit so it does not collide with US state "Georgia"
    'Republic of Georgia': 'Georgia (country)',
    # Common countries seen in this dataset (prevents thousands of geo_country-missing rows)
    'Mongolia': 'Mongolia',
    'Guatemala': 'Guatemala',
    'Malaysia': 'Malaysia',
    'Nigeria': 'Nigeria',
    'Japan': 'Japan',
    'Germany': 'Germany',
    'India': 'India',
    'Belgium': 'Belgium',
    'Spain': 'Spain',
    'United Kingdom': 'United Kingdom',
    'Indonesia': 'Indonesia',
    'Argentina': 'Argentina',
    'South Africa': 'South Africa',
    'Cambodia': 'Cambodia',
    'Brazil': 'Brazil',
    'Australia': 'Australia',
    'Uganda': 'Uganda',
    'Iceland': 'Iceland',
    'Israel': 'Israel',
    'Czech Republic': 'Czech Republic',
    'Pakistan': 'Pakistan',
    'Bhutan': 'Bhutan',
    'Zambia': 'Zambia',
    'Poland': 'Poland',
    'Saudi Arabia': 'Saudi Arabia',
    'Benin': 'Benin',
    'Finland': 'Finland',
    'Canada': 'Canada',
    'Senegal': 'Senegal',
    'Sri Lanka': 'Sri Lanka',
    'Kuwait': 'Kuwait',
}

# Geographic features that should map to states/countries
GEOGRAPHIC_FEATURES = {
    'Interior Alaska': ('Alaska', 'USA'),
    'Southcentral Alaska': ('Alaska', 'USA'),
    'Delaware Bay': ('Delaware', 'USA'),
    'Bay of Plenty': ('Bay of Plenty', 'New Zealand'),
}

# Special administrative regions
SPECIAL_REGIONS = {
    'Hong Kong': ('Hong Kong', 'China'),
}

# Country name standardization
COUNTRY_STANDARDIZATION = {
    'United States': 'USA',
    'England': 'United Kingdom',  # Or keep as England - your choice
    # Disambiguation: keep country explicit so it does not collide with US state "Georgia"
    'Republic of Georgia': 'Georgia (country)',
}

# Location name normalization (handle duplicates/variations/typos)
LOCATION_NORMALIZATION = {
    'Ad Dawhah': 'Ad-Dawhah',  # Standardize to hyphenated version
    'Aguascallientes': 'Aguascalientes',  # Fix typo
    'DC': 'District of Columbia',  # Standardize DC variations
    'DISTRICT OF COLUMBIA': 'District of Columbia',
    'WASHINGTON DC': 'District of Columbia',
    'Viet Nam': 'Vietnam',  # Standardize to single word
    'VIET NAM': 'Vietnam',
    'Bayern': 'Bavaria',
}

# Known acronyms that should stay uppercase
KNOWN_ACRONYMS = ['USA', 'UK', 'USSR', 'UAE', 'CHL', 'AUS', 'VIC', 'NSW', 'QLD', 'WA', 'SA', 'NT', 'ACT', 'TAS']

# How to define the analysis "clean" location column.
# - canonical: keep the canonicalized single token returned by `standardize_location()`
# - city/state/country: collapse to that level when available (fallback to canonical)
# - auto_us_state_else_country: default; for USA use state if available, otherwise country if available
GEO_LOCATION_CLEAN_MODE_DEFAULT = "auto_us_state_else_country"
GEO_LOCATION_CLEAN_MODES = {
    "canonical",
    "city",
    "state",
    "country",
    "auto_us_state_else_country",
}

# Canadian province abbreviations
CANADA_PROVINCE_ABBREV = {
    'NS': 'Nova Scotia', 'NB': 'New Brunswick', 'PE': 'Prince Edward Island',
    'NL': 'Newfoundland and Labrador', 'QC': 'Quebec', 'ON': 'Ontario',
    'MB': 'Manitoba', 'SK': 'Saskatchewan', 'AB': 'Alberta', 'BC': 'British Columbia',
    'YT': 'Yukon', 'NT': 'Northwest Territories', 'NU': 'Nunavut'
}

# German state abbreviations
GERMANY_STATE_ABBREV = {
    'SH': 'Schleswig-Holstein', 'HH': 'Hamburg', 'NI': 'Lower Saxony',
    'HB': 'Bremen', 'NW': 'North Rhine-Westphalia', 'HE': 'Hesse',
    'RP': 'Rhineland-Palatinate', 'BW': 'Baden-Württemberg', 'BY': 'Bavaria',
    'SL': 'Saarland', 'BE': 'Berlin', 'BB': 'Brandenburg', 'MV': 'Mecklenburg-Vorpommern',
    'SN': 'Saxony', 'ST': 'Saxony-Anhalt', 'TH': 'Thuringia'
}

# Mexican state names (common ones)
MEXICO_STATES = {
    'Sonora', 'Chihuahua', 'Baja California', 'Baja California Sur', 'Sinaloa',
    'Nuevo Leon', 'Tamaulipas', 'Coahuila', 'Jalisco', 'Yucatan', 'Quintana Roo'
}


def parse_virus_name(virus_name: str) -> dict[str, Optional[str]]:
    """
    Parse virus name to extract host, location, and year.
    
    Handles patterns like:
    - "Influenza A virus A/Rhode Island/62/2023"
    - "Influenza A virus A/chicken/Hunan/09.03_YYWLP-O12/2021"
    - "Influenza A virus A/swine/Spain/6370-7/2020"
    - "Influenza A virus (A/Virginia/298/2015(H1N1))"
    
    Location parsing notes:
    - Virus name format: "A/location/strain/year" or "A/host/location/strain/year"
    - Only ONE location field is extracted per virus name (parts[1] if host present, parts[0] if no host)
    - If both state and country appear in the name, only the first location part is recorded
    - Some entries may have "USA" as location (country-level) vs specific states like "California"
    - Bird species names (e.g., "ruddy turnstone") are treated as hosts, not locations
    - "environment" is excluded from location (treated as non-location keyword)
    
    Args:
        virus_name: Full virus name string
        
    Returns:
        Dictionary with keys: 'host', 'location', 'year', 'strain'
    """
    result: dict[str, Optional[str]] = {
        "host": None,
        "location": None,
        "year": None,
        "strain": None,
    }

    if not virus_name:
        return result
    if isinstance(virus_name, float) and pd.isna(virus_name):
        return result

    # -----------------------------
    # 1) Extract the core "A/..." payload and split into slash-delimited parts
    # -----------------------------
    core = None
    if isinstance(virus_name, str):
        m = re.search(r"\(A/([^)]*)\)", virus_name)
        if m:
            core = m.group(1)
        elif "A/" in virus_name:
            # Works for both "Influenza A virus A/..." and bare "A/..."
            core = virus_name.split("A/", 1)[1]

    if not core:
        parts = str(virus_name).split("/")
        if len(parts) < 2:
            return result
        parts = parts[1:]
    else:
        parts = core.split("/")

    parts = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    if not parts:
        return result

    # Disambiguation special-case:
    # Some records contain both "(A/.../Georgia/...)" and a trailing "A/.../Republic of Georgia/...".
    # Our current extraction prefers the parenthetical chunk, which can mis-label the country as the US state.
    contains_republic_of_georgia = isinstance(virus_name, str) and ("Republic of Georgia" in virus_name)

    # -----------------------------
    # 2) Helper predicates (keep local to avoid scattering one-off regexes)
    # -----------------------------
    host_keywords = HOST_KEYWORDS
    non_location_keywords = {"environment", "unknown", "na", "n/a", "none", "human"}

    def _lower(s: str) -> str:
        return s.lower().strip()

    def _is_scientific_name(s: str) -> bool:
        # Genus species or Genus_species
        return bool(re.match(r"^[A-Z][a-z]+[\s_][a-z]+", s))

    def _is_scientific_genus(s: str) -> bool:
        s_stripped = s.strip()
        if not s_stripped:
            return False
        if " " in s_stripped or "_" in s_stripped:
            return False
        if not s_stripped[:1].isupper():
            return False
        return _lower(s_stripped) in SCIENTIFIC_GENUS_SET

    def _endswith_bird_animal_term(s: str) -> bool:
        if " " not in s and "-" not in s:
            return False
        words = re.split(r"[\s-]+", _lower(s))
        return len(words) >= 2 and words[-1] in BIRD_ANIMAL_ENDINGS

    def _contains_host_indicator(s: str) -> bool:
        s_lower = _lower(s)
        return (
            any(
                kw.lower() == s_lower
                or re.search(r"\b" + re.escape(kw.lower()) + r"\b", s_lower)
                for kw in host_keywords
            )
            or _is_scientific_name(s)
            or _is_scientific_genus(s)
            or "_" in s
            # Species/common host phrases are typically lowercase multi-word strings:
            # e.g., "wild waterbird", "ring-necked duck"
            or (" " in s and s[:1].islower())
            or _endswith_bird_animal_term(s)
        )

    def _is_non_location_token(s: str) -> bool:
        s_lower = _lower(s)
        return (s_lower in non_location_keywords) or (s_lower in NON_LOCATION_TERMS_SET)

    def _looks_like_strain_id(s: str) -> bool:
        """
        IMPORTANT: require digits.
        This avoids misclassifying plain words like 'Wisconsin' as a strain/ID.
        """
        if not re.search(r"\d", s):
            return False
        s_upper = s.upper()
        return (
            # numeric-heavy patterns with separators
            (re.search(r"\d+[-_]\d+", s) and len(re.findall(r"\d", s)) >= 3)
            or re.match(r"^\d+[-_]", s)
            or ("original" in _lower(s) or "passage" in _lower(s))
            # common alphanumeric ID patterns
            or (len(s) >= 8 and re.match(r"^[A-Z0-9]+$", s_upper))
            or (re.search(r"[A-Z]{3,}", s_upper) and re.search(r"\d{3,}", s))
            or re.match(r"^[A-Z]{1,3}\d{5,}", s_upper)
            or re.match(r"^[A-Z]{2,}\d+[A-Z]+", s_upper)
        )

    def _looks_like_location(s: str) -> bool:
        s_lower = _lower(s)
        s_upper = s.strip().upper()

        # Never treat known non-location tokens (hosts, environmental terms, etc.) as locations
        if _is_non_location_token(s):
            return False

        # Explicit known buckets
        if s_lower in COMMON_MULTIWORD_LOCATIONS or s_lower in COMMON_LOCATIONS:
            return True
        if s_lower in US_STATES_LOWERCASE:
            return True
        if len(s.strip()) == 2 and s_upper in VALID_US_STATE_ABBREVIATIONS:
            return True
        if len(s.strip()) == 2 and s_upper in VALID_US_TERRITORY_ABBREVIATIONS:
            return True
        if s.strip() in COUNTRY_ONLY_LOCATIONS:
            return True
        if s.strip() in CITY_MAPPING or s.strip() in PROVINCE_TO_COUNTRY:
            return True

        # Heuristic: multi-word capitalized locations (e.g., "Western Australia")
        # Exclude obvious host phrases like "... owl", "... duck", "... gull".
        if " " in s and s[0].isupper() and not _endswith_bird_animal_term(s):
            return True

        # Heuristic: single-word capitalized tokens are often locations (states/cities/countries)
        if len(s.split()) == 1 and len(s) > 2 and s[0].isupper() and not _looks_like_strain_id(s):
            return True

        return False

    # -----------------------------
    # 3) Parse year and strain from the right (more stable than branching early)
    # -----------------------------
    year = None
    year_idx = None
    for i in range(len(parts) - 1, -1, -1):
        m = re.search(r"(\d{4})", parts[i])
        if not m:
            continue
        y = int(m.group(1))
        if 1900 <= y <= 2030:
            year = m.group(1)
            year_idx = i
            break
    result["year"] = year

    # Strain is usually immediately left of year, if year exists
    strain_idx = None
    if year_idx is not None and year_idx > 0:
        strain_idx = year_idx - 1
        result["strain"] = parts[strain_idx]

    # Remaining fields on the left hold host/location (and sometimes additional fragments)
    left = parts[:strain_idx] if strain_idx is not None else parts
    if not left:
        return result

    # Skip sentinel non-location leading tokens like "environment"
    while left and len(left) > 1 and _lower(left[0]) in ENVIRONMENTAL_TERMS_SET:
        # e.g. "environment", "water", "air", "pet food", "reassortant"
        left = left[1:]

    # -----------------------------
    # 4) Decide host vs location
    # -----------------------------
    # Default: assume human if we cannot confidently identify a host
    inferred_human = "human"

    if len(left) == 1:
        loc = left[0]
        if not _is_non_location_token(loc):
            result["host"] = inferred_human
            result["location"] = loc
        return result

    first = left[0]
    second = left[1]

    # If first looks like host and second looks like location, treat as host/location
    if _contains_host_indicator(first) and _looks_like_location(second):
        result["host"] = first
        result["location"] = second
        if contains_republic_of_georgia and result["location"] == "Georgia":
            result["location"] = "Republic of Georgia"
        return result

    # Generic pattern: A/<lowercase host-ish>/<LOCATION>/... where host isn't in our lists.
    # Example: guineafowl/India, bobcat/Kansas.
    if first[:1].islower() and _looks_like_location(second):
        result["host"] = first
        result["location"] = second
        if contains_republic_of_georgia and result["location"] == "Georgia":
            result["location"] = "Republic of Georgia"
        return result

    # If first is a known non-location token (e.g., "Pet Food", "Gallus") and second is a location,
    # use the second token as location (host unknown).
    if _is_non_location_token(first) and _looks_like_location(second):
        result["host"] = None
        result["location"] = second
        if contains_republic_of_georgia and result["location"] == "Georgia":
            result["location"] = "Republic of Georgia"
        return result

    # If first looks like location, treat as location-only (host inferred human)
    if _looks_like_location(first):
        result["host"] = inferred_human
        result["location"] = first
        if contains_republic_of_georgia and result["location"] == "Georgia":
            result["location"] = "Republic of Georgia"
        return result

    # Otherwise fall back to location-only with inferred human (unless it's clearly a non-location token)
    if _lower(first) not in non_location_keywords:
        result["host"] = inferred_human
        result["location"] = first
        if contains_republic_of_georgia and result["location"] == "Georgia":
            result["location"] = "Republic of Georgia"

    return result


def load_flu_genomes_key(file_path: Path) -> pd.DataFrame:
    """
    Load and parse the `Flu_Genomes.key` file.
    TODO: should we rename column 'location' to something else so we don't
    conflict with the genomic 'location' column?

    This provides:
      - hash_id, virus_name, hn_subtype
      - seg_1 ... seg_8
      - basic metadata parsed from the name (geo_location_inferred, year, host_inferred)
    """
    print(f"Loading Flu_Genomes.key from: {file_path}")

    # Read the file (tab-separated, no header)
    df = pd.read_csv(file_path, sep='\t', header=None, dtype=str)

    print(f"Loaded {len(df)} rows")
    print(f"Number of columns: {len(df.columns)}")

    # First 3 columns are: hash_id, virus_name, hn_subtype
    df.columns = ['hash_id', 'virus_name', 'hn_subtype'] + [f'seg_{i}' for i in range(1, len(df.columns) - 2)]

    # Parse virus name to extract coarse metadata
    print("Parsing virus names to extract metadata...")
    parsed_data = df['virus_name'].apply(parse_virus_name)

    # Add parsed fields to dataframe
    df['host_inferred'] = [d['host'] for d in parsed_data]
    # IMPORTANT: do not call this column "location" (that word is used elsewhere for genomic locations)
    df['geo_location_inferred'] = [d['location'] for d in parsed_data]
    df['year'] = [d['year'] for d in parsed_data]
    df['strain'] = [d['strain'] for d in parsed_data]

    # Convert year to numeric for analysis
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    return df


def load_flu_first_seg_meta(file_path: Path) -> pd.DataFrame:
    """
    Load and parse the `Flu.first-seg.meta.tab` file.

    Columns (no header in the file; we assign names here):
      1. hash_id
      2. virus_name
      3. hn_subtype
      4. first_seg_id
      5. host_common_name
      6. lab_host
      7. passage
    """
    print(f"Loading Flu.first-seg.meta.tab from: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        dtype=str,
        names=[
            "hash_id",
            "virus_name",
            "hn_subtype",
            "first_seg_id",
            "host_common_name",
            "lab_host",
            "passage",
        ],
    )

    print(f"Loaded {len(df)} rows from Flu.first-seg.meta.tab")
    return df


def standardize_location(location: str) -> dict[str, Optional[str]]:
    """
    Standardize location string and extract City, State, Country components.
    
    Handles:
    - Extracting locations from parentheses (e.g., "X-53A(Puerto Rico" -> "Puerto Rico")
    - Mapping state abbreviations to full names (e.g., "CA" -> "California")
    - Mapping cities to states/countries (e.g., "Chicago" -> Illinois, "Lyon" -> France)
    - Mapping provinces to countries (e.g., "Alberta" -> Canada, "Germany-HE" -> Germany)
    - Filtering out invalid entries (numeric values, parsing errors, non-locations)
    
    Args:
        location: Raw location string (may be None/NaN)
        
    Returns:
        Dictionary with keys: 'geo_location_clean', 'city', 'state', 'country'
        - geo_location_clean: Cleaned geographic location (extracted from parentheses, filtered)
        - city: City name if applicable
        - state: State/province name if applicable
        - country: Country name if applicable
    """
    result = {
        'geo_location_clean': None,
        'city': None,
        'state': None,
        'country': None,
        'reject_reason': None,
    }
    
    if pd.isna(location) or location is None or location == '':
        result["reject_reason"] = "missing_input"
        return result
    
    loc_str = str(location).strip()
    
    # Filter out numeric-only values (parsing errors)
    if loc_str.isdigit() or (len(loc_str) == 4 and loc_str.isdigit() and 1900 <= int(loc_str) <= 2030):
        result["reject_reason"] = "numeric_or_year"
        return result
    
    # Filter out strain identifiers that look like codes (e.g., "23-038138-001-original")
    # Pattern: contains multiple numbers with dashes/underscores, or "original"/"passage" keywords
    if re.search(r'\d+[-_]\d+', loc_str) and len(re.findall(r'\d', loc_str)) >= 3:
        # Has pattern like "23-038138-001" - likely a strain identifier
        if 'original' in loc_str.lower() or 'passage' in loc_str.lower():
            result["reject_reason"] = "strain_like_with_keywords"
            return result
        # If it has 3+ numbers separated by dashes/underscores, it's likely a strain
        if len(re.findall(r'[-_]', loc_str)) >= 2:
            result["reject_reason"] = "strain_like"
            return result
    
    # Filter out alphanumeric codes that look like IDs (e.g., "17OS4133", "AH0003731")
    # Pattern: mix of letters and numbers, typically starting with digits or short alphanumeric codes
    # Allow if it contains spaces (likely multi-word location) or is a known valid abbreviation
    if not ' ' in loc_str and len(loc_str) >= 6:
        # Check if it's mostly alphanumeric with digits (likely a code/ID)
        if re.match(r'^[A-Z0-9]{6,}$', loc_str.upper()):
            # Has both letters and numbers, and no spaces - likely a code
            if re.search(r'[0-9]', loc_str) and re.search(r'[A-Za-z]', loc_str):
                result["reject_reason"] = "id_like"
                return result
    
    # Filter out non-location entries (bird species, animals, parsing errors)
    # Use module-level constant
    loc_lower = loc_str.lower()
    
    # Filter out entries that start with lowercase (almost always non-locations)
    # Exceptions: some valid locations might start with lowercase, but these are rare
    # and usually indicate parsing errors or non-location terms
    if len(loc_str) > 0 and loc_str[0].islower():
        result["reject_reason"] = "starts_with_lowercase"
        return result
    
    # Check for non-location terms (use word boundaries to avoid false positives)
    # Example: "cow" in NON_LOCATION_TERMS should not match "Moscow"
    # Use word boundary matching to ensure we match whole words, not substrings
    for term in NON_LOCATION_TERMS:
        term_lower = term.lower()
        # Use word boundary regex to match whole words only
        # This prevents "cow" from matching "Moscow", "cow" from matching "Scotland", etc.
        if re.search(r'\b' + re.escape(term_lower) + r'\b', loc_lower):
            result["reject_reason"] = f"non_location_term:{term_lower}"
            return result
    
    # Filter out common patterns that indicate non-locations
    # Patterns like "X-XX-XX" (codes), "XX food", "XX milk", etc.
    if re.search(r'\b(food|milk|water|air|env|feces|bioaerosol)\b', loc_lower):
        result["reject_reason"] = "environmental_pattern"
        return result
    
    # Filter out short codes that are likely parsing errors (e.g., "AA", "XX", single letters)
    # Only allow valid US state / territory abbreviations
    if (
        len(loc_str) <= 2
        and loc_str.isalpha()
        and loc_str.upper() not in VALID_US_STATE_ABBREVIATIONS
        and loc_str.upper() not in VALID_US_TERRITORY_ABBREVIATIONS
    ):
        result["reject_reason"] = "short_alpha_not_us_state"
        return result
    
    # Extract location from parentheses (e.g., "X-53A(Puerto Rico" -> "Puerto Rico")
    if '(' in loc_str:
        # Try to extract location from parentheses
        # Pattern: something(location or location-something
        paren_match = re.search(r'\(([^)]+)\)', loc_str)
        if paren_match:
            paren_content = paren_match.group(1)
            # Check if it contains a valid location (not just codes)
            # Look for location-like patterns (words, not just codes)
            if re.search(r'[A-Za-z]{3,}', paren_content):
                # Extract the location part (may be after a dash or code)
                # Try patterns like "X-31B-New Caledonia" or "Puerto Rico"
                if '-' in paren_content:
                    parts = paren_content.split('-')
                    # Take the last part that looks like a location
                    for part in reversed(parts):
                        if re.search(r'^[A-Za-z\s]{3,}', part.strip()):
                            loc_str = part.strip()
                            break
                else:
                    loc_str = paren_content.strip()
        else:
            # Unclosed parenthesis or malformed - try to extract
            if '(' in loc_str and ')' not in loc_str:
                # Pattern like "1931(H1N1" - filter out
                if re.search(r'^\d{4}\(', loc_str) or re.search(r'H\d+N\d+', loc_str):
                    return result
                # Try to extract after opening paren
                after_paren = loc_str.split('(')[-1]
                if re.search(r'[A-Za-z]{3,}', after_paren):
                    loc_str = after_paren.strip()
    
    # Remove any remaining parenthetical content
    loc_str = re.sub(r'\([^)]*\)', '', loc_str).strip()
    
    # Normalize location string (handle whitespace, hyphens, case, etc.)
    # First, normalize whitespace (replace multiple spaces with single, strip)
    loc_str = re.sub(r'\s+', ' ', loc_str).strip()
    # Normalize different types of hyphens/dashes to standard hyphen
    loc_str = re.sub(r'[\u2013\u2014\u2015\u2212]', '-', loc_str)  # en-dash, em-dash, etc. -> hyphen
    
    # Normalize case: convert all-uppercase to title case (e.g., "MEMPHIS" -> "Memphis")
    # This helps with matching city names and other location mappings
    if loc_str.isupper() and len(loc_str) > 2:
        # Convert to title case, but preserve acronyms (2-letter codes like "USA", "UK")
        if len(loc_str) <= 3 or (len(loc_str.split()) == 1 and loc_str.isupper()):
            # Check if it's a known acronym/abbreviation that should stay uppercase
            if loc_str not in KNOWN_ACRONYMS:
                loc_str = loc_str.title()
        else:
            # Multi-word all-uppercase -> title case
            loc_str = loc_str.title()
    
    # Apply location name normalization (case-insensitive for variations)
    loc_upper = loc_str.upper()
    if loc_str in LOCATION_NORMALIZATION:
        loc_str = LOCATION_NORMALIZATION[loc_str]
    elif loc_upper in [k.upper() for k in LOCATION_NORMALIZATION.keys()]:
        # Case-insensitive match for normalization keys
        for key, value in LOCATION_NORMALIZATION.items():
            if key.upper() == loc_upper:
                loc_str = value
                break

    # Optional: if `pycountry` is installed, recognize country names/aliases without hardcoding.
    pycountry = None
    if importlib.util.find_spec("pycountry") is not None:  # pragma: no cover
        try:
            pycountry = importlib.import_module("pycountry")
        except Exception:
            pycountry = None

    if pycountry is not None:  # pragma: no cover
        try:
            c = pycountry.countries.lookup(loc_str)
            cname = c.name
            if cname in COUNTRY_STANDARDIZATION:
                cname = COUNTRY_STANDARDIZATION[cname]
            result.update({'geo_location_clean': loc_str, 'country': cname})
            return result
        except Exception:
            pass
    
    # Initialize
    city = None
    state = None
    country = None
    
    # Check if it's a US state abbreviation
    loc_upper = loc_str.upper()
    if loc_upper in US_STATE_ABBREV:
        state = US_STATE_ABBREV[loc_upper]
        country = 'USA'
        result.update({
            'geo_location_clean': state,  # Normalize to full state name
            'state': state,
            'country': country
        })
        return result

    # Check if it's a US territory abbreviation (e.g., PR, GU)
    if loc_upper in US_TERRITORY_ABBREV:
        state = US_TERRITORY_ABBREV[loc_upper]
        country = 'USA'
        result.update({
            'geo_location_clean': state,
            'state': state,
            'country': country
        })
        return result
    
    # Check if it's a city that should map to state/country
    # First check for compound names like "Rockville Illinois"
    city_found = None
    for city_key in sorted(CITY_MAPPING.keys(), key=len, reverse=True):  # Check longer names first
        if city_key in loc_str:
            city_found = city_key
            break
    
    if city_found:
        mapped_location = CITY_MAPPING[city_found]
        # Determine if mapped location is a state or country
        if mapped_location in US_STATE_ABBREV.values() or mapped_location in ['District of Columbia']:
            city = city_found
            state = mapped_location
            country = 'USA'
            # Normalize geo_location_clean to state name for consistency
            geo_location_clean = state
        else:
            city = city_found
            # Standardize country-like values returned from CITY_MAPPING (e.g., England -> United Kingdom)
            country = COUNTRY_ONLY_LOCATIONS.get(mapped_location, mapped_location)
            country = COUNTRY_STANDARDIZATION.get(country, country)
            # For international cities, keep city name
            geo_location_clean = city_found
        result.update({
            'geo_location_clean': geo_location_clean,
            'city': city,
            'state': state,
            'country': country
        })
        return result
    
    # Check if it's a province/state abbreviation (e.g., ALB -> Alberta)
    loc_upper = loc_str.upper()
    if loc_upper in PROVINCE_ABBREV:
        full_name = PROVINCE_ABBREV[loc_upper]
        # Map to country if known
        if full_name in PROVINCE_TO_COUNTRY:
            state = full_name
            country = PROVINCE_TO_COUNTRY[full_name]
            result.update({
                'geo_location_clean': full_name,  # Use full name
                'state': state,
                'country': country
            })
            return result
        else:
            # Just use the full name
            result.update({
                'geo_location_clean': full_name,
                'state': full_name
            })
            return result
    
    # Check if it's a geographic feature that should map to state/country
    if loc_str in GEOGRAPHIC_FEATURES:
        state, country = GEOGRAPHIC_FEATURES[loc_str]
        result.update({
            'geo_location_clean': loc_str,  # Keep original name (e.g., "Interior Alaska")
            'state': state,
            'country': country
        })
        return result
    
    # Check if it's a special administrative region
    if loc_str in SPECIAL_REGIONS:
        state, country = SPECIAL_REGIONS[loc_str]
        result.update({
            'geo_location_clean': loc_str,
            'state': state,
            'country': country
        })
        return result
    
    # Check if it's a province that should map to country
    if loc_str in PROVINCE_TO_COUNTRY:
        state = loc_str
        country = PROVINCE_TO_COUNTRY[loc_str]
        result.update({
            'geo_location_clean': loc_str,
            'state': state,
            'country': country
        })
        return result
    
    # Check if it's a country-only location (no state/province)
    if loc_str in COUNTRY_ONLY_LOCATIONS:
        country = COUNTRY_ONLY_LOCATIONS[loc_str]
        result.update({
            'geo_location_clean': loc_str,
            'country': country
        })
        return result
    
    # Handle hyphenated country-state/province entries (e.g., "Mexico-Sonora", "Canada-NS", "Germany-SH")
    # Parse these early to extract country and state/province
    if '-' in loc_str and not loc_str.startswith('Interior '):  # Exclude "Interior Alaska" which is handled separately
        parts = loc_str.split('-', 1)  # Split on first hyphen only
        country_part = parts[0].strip()
        state_part = parts[1].strip() if len(parts) > 1 else None
        
        # Parse based on country
        if country_part == 'Canada' and state_part:
            if state_part.upper() in CANADA_PROVINCE_ABBREV:
                state = CANADA_PROVINCE_ABBREV[state_part.upper()]
                country = 'Canada'
                result.update({
                    'geo_location_clean': state,  # Use full province name
                    'state': state,
                    'country': country
                })
                return result
        elif country_part == 'Germany' and state_part:
            if state_part.upper() in GERMANY_STATE_ABBREV:
                state = GERMANY_STATE_ABBREV[state_part.upper()]
                country = 'Germany'
                result.update({
                    'geo_location_clean': state,  # Use full state name
                    'state': state,
                    'country': country
                })
                return result
            else:
                # Unknown German state abbreviation, keep as-is
                country = 'Germany'
                state = loc_str  # Keep full name like "Germany-HE"
                result.update({
                    'geo_location_clean': loc_str,
                    'state': state,
                    'country': country
                })
                return result
        elif country_part == 'Mexico' and state_part:
            # Check if state_part is a known Mexican state
            if state_part in MEXICO_STATES or state_part.title() in MEXICO_STATES:
                state = state_part.title() if state_part not in MEXICO_STATES else state_part
                country = 'Mexico'
                result.update({
                    'geo_location_clean': state,
                    'state': state,
                    'country': country
                })
                return result
            else:
                # Unknown Mexican state, keep as-is but set country
                country = 'Mexico'
                state = state_part
                result.update({
                    'geo_location_clean': loc_str,  # Keep full "Mexico-State" format
                    'state': state,
                    'country': country
                })
                return result

        # Generic fallback: sometimes tokens look like "SOME_PREFIX-Uganda".
        # If the suffix is a known country, treat it as country-level.
        if state_part and state_part in COUNTRY_ONLY_LOCATIONS:
            country = COUNTRY_ONLY_LOCATIONS[state_part]
            result.update({
                'geo_location_clean': state_part,
                'country': country
            })
            return result
    
    # Handle German states (Germany-XX -> Germany) - fallback for unhandled cases
    if loc_str.startswith('Germany-'):
        country = 'Germany'
        state = loc_str  # Keep full name like "Germany-HE"
        result.update({
            'geo_location_clean': loc_str,
            'state': state,
            'country': country
        })
        return result
    
    # Handle District of Columbia (already normalized by location_normalization)
    if loc_str == 'District of Columbia':
        state = 'District of Columbia'
        country = 'USA'
        result.update({
            'geo_location_clean': loc_str,
            'state': state,
            'country': country
        })
        return result
    
    # Standardize country names
    if loc_str in COUNTRY_STANDARDIZATION:
        country = COUNTRY_STANDARDIZATION[loc_str]
        result.update({
            'geo_location_clean': loc_str,
            'country': country
        })
        return result
    
    # Check if it's a known US state (full name)
    if loc_str in US_STATE_ABBREV.values() or loc_str == 'District of Columbia':
        state = loc_str
        country = 'USA'
        result.update({
            'geo_location_clean': loc_str,
            'state': state,
            'country': country
        })
        return result
    
    # Default: keep as-is, try to infer country from common patterns
    # This is a simplified version - you might want to add more logic
    result.update({
        'geo_location_clean': loc_str
    })
    
    return result


def select_geo_location_clean(
    *,
    canonical: Optional[str],
    city: Optional[str],
    state: Optional[str],
    country: Optional[str],
    mode: str = GEO_LOCATION_CLEAN_MODE_DEFAULT,
) -> Optional[str]:
    """
    Choose what `geo_location_clean` should represent for downstream analysis.

    Important:
    - We only ever extract one location token from `virus_name`. That means we cannot reliably
      produce sub-country admin levels like "county" unless the token itself is a county name
      and we have a mapping for it (we don't today).
    - If you want the historical behavior, use mode="canonical" and rely on `geo_location_canonical`.
    """
    if mode not in GEO_LOCATION_CLEAN_MODES:
        raise ValueError(f"Unknown geo_location_clean mode: {mode!r}. Allowed: {sorted(GEO_LOCATION_CLEAN_MODES)}")

    if mode == "canonical":
        return canonical
    if mode == "city":
        return city or canonical
    if mode == "state":
        return state or canonical
    if mode == "country":
        return country or canonical

    # Default policy: if USA, use state; otherwise use country.
    # Fallback to canonical if we couldn't infer state/country.
    if mode == "auto_us_state_else_country":
        if country == "USA":
            return state or canonical or country
        return country or canonical

    # Should be unreachable because of the guard above
    return canonical


def build_analysis_dataframe(
    key_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    geo_location_clean_mode: str = GEO_LOCATION_CLEAN_MODE_DEFAULT,
    ) -> pd.DataFrame:
    """
    Merge key and metadata tables and construct analysis-ready columns.
    The merge brings passage information into key_df from meta_df.

    Available columns in key_df:
    - hash_id
    - virus_name
    - hn_subtype
    - seg_1 ... seg_8
    - host_inferred (inferred from virus name)
    - geo_location_inferred (inferred from virus name)
    - year (inferred from virus name)
    - strain (inferred from virus name)

    Available columns in meta_df:
    - hash_id
    - virus_name
    - hn_subtype
    - first_seg_id
    - host_common_name
    - lab_host
    - passage

    The dataframes have 3 columns in common: hash_id, virus_name, hn_subtype.
    - Left-join on hash_id so we keep the full set of genomes from
      `Flu_Genomes.key` even if a few are missing from the meta file.
    - Prefer explicit host information from `host_common_name`. If not available,
      fall back to inferred host from `host_inferred`.
    - Keep lab_host and passage as separate sparse fields.
    """
    print("\nMerging Flu_Genomes.key with Flu.first-seg.meta.tab ...")

    merged = key_df.merge(
        meta_df,
        on=["hash_id"],
        how="left",
        suffixes=("", "_meta"),
    )

    # Sanity check: how many rows lack metadata?
    n_missing_meta = merged["host_common_name"].isna().sum()
    print(
        f"Rows without metadata in Flu.first-seg.meta.tab: "
        f"{n_missing_meta} ({100 * n_missing_meta / len(merged):.2f}%)"
    )

    # For host, prefer explicit host_common_name and fall back to inferred host
    merged["host"] = merged["host_common_name"].fillna(merged["host_inferred"])

    # Normalize host values to avoid trivial duplicates like "Human" vs "human"
    def _normalize_host_value(x: object) -> Optional[str]:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "null":
            return None
        s_lower = s.lower()
        # Canonicalize a few high-impact values explicitly
        if s_lower == "human":
            return "Human"
        if s_lower == "environment":
            return "Environment"
        # If it's entirely lowercase, treat it as a common-name host and title-case it
        # (keeps existing capitalization for already-normalized values like "Great-Horned Owl")
        if s.islower():
            return s.title()
        return s

    merged["host"] = merged["host"].apply(_normalize_host_value)

    # Ensure year is numeric, then cast to int (NaN values remain as NaN/float)
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce")
    # Convert to Int64 (nullable integer) to preserve NaN values
    merged["year"] = merged["year"].astype("Int64")
    
    # Apply location standardization
    print("\nStandardizing locations...")
    location_results = merged["geo_location_inferred"].apply(standardize_location)
    
    # Extract standardized location components
    canonical = [d.get("geo_location_clean") for d in location_results]
    geo_city = [d.get("city") for d in location_results]
    geo_state = [d.get("state") for d in location_results]
    geo_country = [d.get("country") for d in location_results]
    geo_reject_reason = [d.get("reject_reason") for d in location_results]

    # Keep the canonical standardized token (historical behavior) separate from the analysis "clean" value
    merged["geo_location_canonical"] = canonical
    merged["geo_city"] = geo_city
    merged["geo_state"] = geo_state
    merged["geo_country"] = geo_country
    merged["geo_location_reject_reason"] = geo_reject_reason

    merged["geo_location_clean"] = [
        select_geo_location_clean(
            canonical=canonical[i],
            city=geo_city[i],
            state=geo_state[i],
            country=geo_country[i],
            mode=geo_location_clean_mode,
        )
        for i in range(len(merged))
    ]
    
    # Preserve original inferred location as geo_location_raw, then remove inferred column
    merged["geo_location_raw"] = merged["geo_location_inferred"].copy()
    # Remove inferred column to avoid confusion with genomic 'location' column in protein DataFrames
    merged = merged.drop(columns=["geo_location_inferred"])

    return merged


def analyze_metadata(df: pd.DataFrame) -> None:
    """
    Perform exploratory analysis on the metadata.
    
    Args:
        df: DataFrame with parsed metadata
    """
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Basic statistics
    print(f"\nTotal isolates: {len(df)}")
    print(f"Unique hash IDs: {df['hash_id'].nunique()}")
    print(f"Unique virus names: {df['virus_name'].nunique()}")
    
    # Missing values in key analysis fields
    print("\n--- Missing Values ---")
    missing = df[["host", "geo_location_clean", "year", "hn_subtype"]].isnull().sum()
    for col, count in missing.items():
        pct = 100 * count / len(df)
        print(f"{col}: {count} ({pct:.1f}%)")
    
    # Host organism distribution
    print("\n--- Host Organism Distribution ---")
    host_counts = df["host"].value_counts(dropna=False)
    print(host_counts)
    print(f"\nTotal unique hosts: {df['host'].nunique()}")

    # Lab host (non-empty only, to see how sparse it is)
    print("\n--- Lab Host (non-empty, top 10) ---")
    lab_nonempty = df["lab_host"].replace("", pd.NA).dropna()
    if not lab_nonempty.empty:
        print(lab_nonempty.value_counts().head(10))
    else:
        print("No lab_host values present.")

    # Passage info (non-empty only)
    print("\n--- Passage (non-empty, top 10) ---")
    passage_nonempty = df["passage"].replace("", pd.NA).dropna()
    if not passage_nonempty.empty:
        print(passage_nonempty.value_counts().head(10))
    else:
        print("No passage values present.")
    
    # H/N subtype distribution
    print("\n--- H/N Subtype Distribution ---")
    subtype_counts = df["hn_subtype"].value_counts(dropna=False)
    print(subtype_counts)
    print(f"\nTotal unique subtypes: {df['hn_subtype'].nunique()}")
    
    # Year distribution
    print("\n--- Year Distribution ---")
    year_counts = df["year"].value_counts().sort_index()
    print(f"Year range: {df['year'].min():.0f} - {df['year'].max():.0f}")
    print(f"Years with data: {df['year'].notna().sum()}")
    print("\nTop 10 years by count:")
    print(year_counts.tail(10))
    
    # Location distribution (top locations)
    print("\n--- Top 20 Locations ---")
    location_counts = df["geo_location_clean"].value_counts(dropna=False).head(20)
    print(location_counts)
    
    # Cross-tabulation: Host vs H/N subtype
    print("\n--- Host vs H/N Subtype Cross-tabulation ---")
    crosstab = pd.crosstab(df["host"], df["hn_subtype"], margins=True)
    print(crosstab)
    
    # Year vs H/N subtype
    print("\n--- Year vs H/N Subtype (sample) ---")
    year_subtype = pd.crosstab(df["year"], df["hn_subtype"], margins=True)
    print(year_subtype.tail(15))  # Show last 15 years
    
    # Segment count analysis
    segment_cols = [col for col in df.columns if col.startswith("seg_")]
    print(f"\n--- Segment Information ---")
    print(f"Number of segment columns: {len(segment_cols)}")
    if segment_cols:
        # Count non-null segments per row
        df['n_segments'] = df[segment_cols].notna().sum(axis=1)
        print(f"Segments per isolate:")
        print(df['n_segments'].value_counts().sort_index())


def identify_data_quality_issues(df: pd.DataFrame) -> None:
    """
    Identify potential data quality issues.
    
    Args:
        df: DataFrame with parsed metadata
    """
    print("\n" + "="*80)
    print("DATA QUALITY CHECKS")
    print("="*80)
    
    issues = []
    
    # Check for duplicate hash IDs
    duplicates = df["hash_id"].duplicated()
    if duplicates.any():
        n_dup = duplicates.sum()
        issues.append(f"Duplicate hash IDs: {n_dup}")
        print(f"⚠️  Found {n_dup} duplicate hash IDs")
        print(
            df[df["hash_id"].duplicated(keep=False)][
                ["hash_id", "virus_name"]
            ].head(10)
        )
    
    # Check for missing critical fields
    missing_host = df["host"].isna().sum()
    if missing_host > 0:
        issues.append(f"Missing host: {missing_host}")
        print(f"⚠️  Found {missing_host} isolates with missing host")
    
    missing_year = df["year"].isna().sum()
    if missing_year > 0:
        issues.append(f"Missing year: {missing_year}")
        print(f"⚠️  Found {missing_year} isolates with missing year")
    
    missing_subtype = df["hn_subtype"].isna().sum()
    if missing_subtype > 0:
        issues.append(f"Missing H/N subtype: {missing_subtype}")
        print(f"⚠️  Found {missing_subtype} isolates with missing H/N subtype")
    
    # Check for unusual years
    if df["year"].notna().any():
        min_year = df["year"].min()
        max_year = df["year"].max()
        if min_year < 1900 or max_year > 2030:
            issues.append(f"Unusual year range: {min_year:.0f} - {max_year:.0f}")
            print(f"⚠️  Unusual year range: {min_year:.0f} - {max_year:.0f}")
    
    # Check for empty virus names
    empty_names = (
        df["virus_name"].isna() | (df["virus_name"].str.strip() == "")
    ).sum()
    if empty_names > 0:
        issues.append(f"Empty virus names: {empty_names}")
        print(f"⚠️  Found {empty_names} isolates with empty virus names")
    
    if not issues:
        print("✓ No major data quality issues detected")
    else:
        print(f"\nTotal issues found: {len(issues)}")


def save_summary_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save summary statistics to files.
    
    Args:
        df: DataFrame with parsed metadata
        output_dir: Directory to save output files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full parsed dataset
    output_file = output_dir / "flu_genomes_metadata_parsed.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved parsed metadata to: {output_file}")
    
    # Save summary statistics
    summary_file = output_dir / "flu_genomes_metadata_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Flu Genomes Metadata - Summary Statistics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total isolates: {len(df)}\n")
        f.write(f"Unique hash IDs: {df['hash_id'].nunique()}\n")
        f.write(f"Unique virus names: {df['virus_name'].nunique()}\n\n")

        f.write("Host Distribution:\n")
        f.write(str(df["host"].value_counts(dropna=False)) + "\n\n")

        f.write("H/N Subtype Distribution:\n")
        f.write(str(df["hn_subtype"].value_counts(dropna=False)) + "\n\n")

        f.write("Location Distribution:\n")
        f.write(str(df["geo_location_clean"].value_counts(dropna=False)) + "\n\n")

        f.write("Passage Distribution:\n")
        # Treat empty strings as missing for passage
        df_passage = df["passage"].replace("", pd.NA)
        f.write(str(df_passage.value_counts(dropna=False)) + "\n\n")

        if df["year"].notna().any():
            f.write(
                "Year Range: {:.0f} - {:.0f}\n".format(
                    df["year"].min(), df["year"].max()
                )
            )
        else:
            f.write("Year Range: N/A\n")
        f.write(f"Years with data: {df['year'].notna().sum()}\n\n")
    
    print(f"Saved summary statistics to: {summary_file}")
    
    # Save individual count CSV files
    save_metadata_counts_csv(df, output_dir)


def save_metadata_counts_csv(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save metadata count distributions as CSV files.
    
    Creates separate CSV files for each metadata field:
    - host_count.csv
    - subtype_count.csv (hn_subtype)
    - geo_location_count.csv
    - passage_count.csv
    
    Each CSV has two columns: value and count, sorted by count (descending).
    
    Args:
        df: DataFrame with parsed metadata
        output_dir: Directory to save output files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Host count
    host_counts = df["host"].value_counts(dropna=False).reset_index()
    host_counts.columns = ["host", "count"]
    host_counts = host_counts.sort_values("count", ascending=False)
    host_file = output_dir / "host_count.csv"
    host_counts.to_csv(host_file, index=False)
    print(f"Saved: {host_file}")
    
    # Subtype count
    subtype_counts = df["hn_subtype"].value_counts(dropna=False).reset_index()
    subtype_counts.columns = ["hn_subtype", "count"]
    subtype_counts = subtype_counts.sort_values("count", ascending=False)
    subtype_file = output_dir / "subtype_count.csv"
    subtype_counts.to_csv(subtype_file, index=False)
    print(f"Saved: {subtype_file}")
    
    # Location count (uses cleaned location from build_analysis_dataframe)
    # Note: df["geo_location_clean"] contains the cleaned location
    location_counts = df["geo_location_clean"].value_counts(dropna=False).reset_index()
    location_counts.columns = ["geo_location_clean", "count"]
    location_counts = location_counts.sort_values("count", ascending=False)
    location_file = output_dir / "geo_location_count.csv"
    location_counts.to_csv(location_file, index=False)
    print(f"Saved: {location_file}")
    
    # Passage count (treat empty strings as missing)
    df_passage = df["passage"].replace("", pd.NA)
    passage_counts = df_passage.value_counts(dropna=False).reset_index()
    passage_counts.columns = ["passage", "count"]
    passage_counts = passage_counts.sort_values("count", ascending=False)
    passage_file = output_dir / "passage_count.csv"
    passage_counts.to_csv(passage_file, index=False)
    print(f"Saved: {passage_file}")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_temporal_by_subtype(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10) -> None:
    """
    Plot temporal distribution by H/N subtype (stacked area plot).
    
    Shows how different subtypes have changed in prevalence over time.
    """
    # Filter to rows with valid year and subtype, and start from 1990
    df_plot = df[(df["year"].notna()) & (df["hn_subtype"].notna()) & (df["year"] >= 1990)].copy()
    
    # Get top N subtypes by total count
    top_subtypes = df_plot["hn_subtype"].value_counts().head(top_n).index.tolist()
    df_plot = df_plot[df_plot["hn_subtype"].isin(top_subtypes)]
    
    # Count by year and subtype
    year_subtype_counts = (
        df_plot.groupby(["year", "hn_subtype"])
        .size()
        .reset_index(name="count")
        .pivot(index="year", columns="hn_subtype", values="count")
        .fillna(0)
    )
    
    # Sort by year
    year_subtype_counts = year_subtype_counts.sort_index()
    
    # Create smoother curve by interpolating to more points
    # Generate a continuous year range (every 0.1 year for smoothness)
    min_year = int(year_subtype_counts.index.min())
    max_year = int(year_subtype_counts.index.max())
    smooth_years = np.arange(min_year, max_year + 1, 0.1)
    
    # Interpolate each subtype's data
    year_subtype_counts_smooth = pd.DataFrame(index=smooth_years)
    for col in top_subtypes:
        # Use cubic interpolation for smooth curves
        f = interp1d(
            year_subtype_counts.index.values,
            year_subtype_counts[col].values,
            kind='cubic',
            bounds_error=False,
            fill_value=0
        )
        year_subtype_counts_smooth[col] = f(smooth_years)
        # Ensure no negative values
        year_subtype_counts_smooth[col] = np.maximum(year_subtype_counts_smooth[col], 0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.stackplot(
        year_subtype_counts_smooth.index,
        *[year_subtype_counts_smooth[col] for col in top_subtypes],
        labels=top_subtypes,
        alpha=0.7
    )
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Number of Isolates", fontsize=13)
    ax.set_title(f"Temporal Distribution by H/N Subtype (Top {top_n})", fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', ncol=2, fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "temporal_distribution_by_subtype.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_temporal_by_host(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10) -> None:
    """
    Plot temporal distribution by host (stacked area plot).
    
    Shows how different hosts have changed in prevalence over time.
    """
    # Filter to rows with valid year and host, and start from 1990
    df_plot = df[(df["year"].notna()) & (df["host"].notna()) & (df["year"] >= 1990)].copy()
    
    # Get top N hosts by total count
    top_hosts = df_plot["host"].value_counts().head(top_n).index.tolist()
    df_plot = df_plot[df_plot["host"].isin(top_hosts)]
    
    # Count by year and host
    year_host_counts = (
        df_plot.groupby(["year", "host"])
        .size()
        .reset_index(name="count")
        .pivot(index="year", columns="host", values="count")
        .fillna(0)
    )
    
    # Sort by year
    year_host_counts = year_host_counts.sort_index()
    
    # Create smoother curve by interpolating to more points
    # Generate a continuous year range (every 0.1 year for smoothness)
    min_year = int(year_host_counts.index.min())
    max_year = int(year_host_counts.index.max())
    smooth_years = np.arange(min_year, max_year + 1, 0.1)
    
    # Interpolate each host's data
    year_host_counts_smooth = pd.DataFrame(index=smooth_years)
    for col in top_hosts:
        # Use cubic interpolation for smooth curves
        f = interp1d(
            year_host_counts.index.values,
            year_host_counts[col].values,
            kind='cubic',
            bounds_error=False,
            fill_value=0
        )
        year_host_counts_smooth[col] = f(smooth_years)
        # Ensure no negative values
        year_host_counts_smooth[col] = np.maximum(year_host_counts_smooth[col], 0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.stackplot(
        year_host_counts_smooth.index,
        *[year_host_counts_smooth[col] for col in top_hosts],
        labels=top_hosts,
        alpha=0.7
    )
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Number of Isolates", fontsize=13)
    ax.set_title(f"Temporal Distribution by Host (Top {top_n})", fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', ncol=2, fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "temporal_distribution_by_host.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_temporal_combined(
    df: pd.DataFrame,
    output_dir: Path,
    top_subtypes: int = 5,
    top_hosts: int = 5) -> None:
    """
    Combined temporal plot showing both subtype and host trends.
    
    Uses two subplots: one for subtypes, one for hosts.
    """
    # Filter to rows with valid year, and start from 1990
    df_plot = df[(df["year"].notna()) & (df["year"] >= 1990)].copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Top plot: Subtypes
    if df_plot["hn_subtype"].notna().any():
        top_subtype_list = df_plot["hn_subtype"].value_counts().head(top_subtypes).index.tolist()
        df_subtype = df_plot[df_plot["hn_subtype"].isin(top_subtype_list)]
        
        year_subtype_counts = (
            df_subtype.groupby(["year", "hn_subtype"])
            .size()
            .reset_index(name="count")
            .pivot(index="year", columns="hn_subtype", values="count")
            .fillna(0)
            .sort_index()
        )
        
        # Create smoother curve by interpolating
        min_year = int(year_subtype_counts.index.min())
        max_year = int(year_subtype_counts.index.max())
        smooth_years = np.arange(min_year, max_year + 1, 0.1)
        
        year_subtype_counts_smooth = pd.DataFrame(index=smooth_years)
        for col in top_subtype_list:
            f = interp1d(
                year_subtype_counts.index.values,
                year_subtype_counts[col].values,
                kind='cubic',
                bounds_error=False,
                fill_value=0
            )
            year_subtype_counts_smooth[col] = np.maximum(f(smooth_years), 0)
        
        ax1.stackplot(
            year_subtype_counts_smooth.index,
            *[year_subtype_counts_smooth[col] for col in top_subtype_list],
            labels=top_subtype_list,
            alpha=0.7
        )
        ax1.set_ylabel("Number of Isolates", fontsize=12)
        ax1.set_title(f"Temporal Distribution by H/N Subtype (Top {top_subtypes})", fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', ncol=2, fontsize=9)
        ax1.grid(alpha=0.3)
    
    # Bottom plot: Hosts
    if df_plot["host"].notna().any():
        top_host_list = df_plot["host"].value_counts().head(top_hosts).index.tolist()
        df_host = df_plot[df_plot["host"].isin(top_host_list)]
        
        year_host_counts = (
            df_host.groupby(["year", "host"])
            .size()
            .reset_index(name="count")
            .pivot(index="year", columns="host", values="count")
            .fillna(0)
            .sort_index()
        )
        
        # Create smoother curve by interpolating
        min_year = int(year_host_counts.index.min())
        max_year = int(year_host_counts.index.max())
        smooth_years = np.arange(min_year, max_year + 1, 0.1)
        
        year_host_counts_smooth = pd.DataFrame(index=smooth_years)
        for col in top_host_list:
            f = interp1d(
                year_host_counts.index.values,
                year_host_counts[col].values,
                kind='cubic',
                bounds_error=False,
                fill_value=0
            )
            year_host_counts_smooth[col] = np.maximum(f(smooth_years), 0)
        
        ax2.stackplot(
            year_host_counts_smooth.index,
            *[year_host_counts_smooth[col] for col in top_host_list],
            labels=top_host_list,
            alpha=0.7
        )
        ax2.set_xlabel("Year", fontsize=12)
        ax2.set_ylabel("Number of Isolates", fontsize=12)
        ax2.set_title(f"Temporal Distribution by Host (Top {top_hosts})", fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left', ncol=2, fontsize=9)
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "temporal_distribution_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_host_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20) -> None:
    """
    Plot host distribution as horizontal bar chart (top N hosts).
    
    Uses blue color scheme to distinguish from subtype plot.
    """
    # Get total unique hosts for title
    total_hosts = df["host"].notna().sum()
    unique_hosts = df["host"].nunique()
    
    host_counts = df["host"].value_counts(dropna=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    
    # Use blue color scheme
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(host_counts)))
    ax.barh(range(len(host_counts)), host_counts.values, color=colors)
    ax.set_yticks(range(len(host_counts)))
    ax.set_yticklabels(host_counts.index, fontsize=10)
    ax.set_xlabel("Number of Isolates", fontsize=13)
    ax.set_title(f"Host Distribution (Top {top_n} out of {unique_hosts} available)", fontsize=15, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.grid(axis='y', visible=False)  # Explicitly disable horizontal grid lines
    
    # Add value labels on bars
    for i, v in enumerate(host_counts.values):
        ax.text(v + max(host_counts.values) * 0.01, i, f'{v:,}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "host_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_subtype_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20) -> None:
    """
    Plot H/N subtype distribution as horizontal bar chart (top N subtypes).
    
    Uses red/orange color scheme and different texture to distinguish from host plot.
    Uses same top_n as host_distribution for consistency.
    """
    # Get total unique subtypes for title
    unique_subtypes = df["hn_subtype"].nunique()
    
    subtype_counts = df["hn_subtype"].value_counts(dropna=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    
    # Use red/orange color scheme with different pattern
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(subtype_counts)))
    bars = ax.barh(range(len(subtype_counts)), subtype_counts.values, color=colors, 
                   edgecolor='black', linewidth=1.5)
    # Add diagonal hatch pattern to distinguish from host plot (black for visibility)
    for bar in bars:
        bar.set_hatch('///')
    ax.set_yticks(range(len(subtype_counts)))
    ax.set_yticklabels(subtype_counts.index, fontsize=10)
    ax.set_xlabel("Number of Isolates", fontsize=13)
    ax.set_title(f"H/N Subtype Distribution (Top {top_n} out of {unique_subtypes} available)", fontsize=15, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.grid(axis='y', visible=False)  # Explicitly disable horizontal grid lines
    
    # Add value labels on bars
    for i, v in enumerate(subtype_counts.values):
        ax.text(v + max(subtype_counts.values) * 0.01, i, f'{v:,}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "subtype_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_location_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20) -> None:
    """
    Plot geographic location distribution as horizontal bar chart (top N locations).
    
    Uses green color scheme to distinguish from other plots.
    """
    # Only show top N non-missing locations (do NOT force a "Missing" bar into the plot)
    all_non_missing_counts = df["geo_location_clean"].value_counts(dropna=True)
    unique_locations = len(all_non_missing_counts)
    location_counts = all_non_missing_counts.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(location_counts) * 0.3)))
    
    # Use green color scheme
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(location_counts)))
    
    display_labels = [str(idx) for idx in location_counts.index]
    
    bars = ax.barh(range(len(location_counts)), location_counts.values, color=colors)
    ax.set_yticks(range(len(location_counts)))
    ax.set_yticklabels(display_labels, fontsize=10)
    ax.set_xlabel("Number of Isolates", fontsize=13)
    
    title = f"Geographic Location Distribution (Top {len(location_counts)} out of {unique_locations} available)"
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.grid(axis='y', visible=False)
    
    # Add value labels on bars
    for i, v in enumerate(location_counts.values):
        ax.text(v + max(location_counts.values) * 0.01, i, f'{v:,}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "geo_location_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_passage_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20) -> None:
    """
    Plot passage distribution as horizontal bar chart (top N passages).
    
    Passage field comes from Flu.first-seg.meta.tab (column 7) and contains
    passage history information as strings. Common values:
    - "Original" or "Primary Specimen": Direct from original host (no lab passage)
    - "S1", "S2", etc.: Sequential passages (S1 = first passage, S2 = second, etc.)
    - "Egg", "Egg1", etc.: Passaged in eggs (Egg1 = first egg passage)
    - "MDCK": Passaged in MDCK cell line (Madin-Darby Canine Kidney cells)
    
    Note: "Primary Specimen" and "Original" both refer to unpassaged samples,
    but may be recorded differently in the source data.
    
    Includes missing/empty values in the plot.
    Uses purple color scheme to distinguish from other plots.
    """
    # Count missing/empty values (treat empty strings as missing)
    df_passage = df["passage"].replace("", pd.NA)
    missing_count = df_passage.isna().sum()
    
    # Get ALL value counts EXCLUDING missing (dropna=True to avoid NaN in index)
    all_non_missing_counts = df_passage.value_counts(dropna=True)
    unique_passages = len(all_non_missing_counts)  # Total unique non-missing passages
    
    # Get top_n non-missing values
    non_missing_counts = all_non_missing_counts.head(top_n)
    has_missing = missing_count > 0
    
    # If we have missing values, always include them as a single "Missing" entry
    if has_missing:
        # Create a combined series with missing
        missing_series = pd.Series([missing_count], index=["Missing"])
        # Combine and sort by count
        combined = pd.concat([non_missing_counts, missing_series]).sort_values(ascending=False)
        # Take exactly top_n items (if "Missing" is in top_n, it will be included)
        # We need to ensure we don't exceed top_n total items
        if "Missing" in combined.head(top_n).index:
            # "Missing" is already in top_n, so just take top_n
            passage_counts = combined.head(top_n)
        else:
            # "Missing" is not in top_n, so we need to include it
            # Take top_n-1 non-missing + "Missing" = top_n total
            passage_counts = pd.concat([non_missing_counts.head(top_n - 1), missing_series]).sort_values(ascending=False)
    else:
        passage_counts = non_missing_counts
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(passage_counts) * 0.3)))
    
    # Use purple color scheme
    colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(passage_counts)))
    
    # Get display labels (handle "Missing" specially)
    display_labels = []
    for idx in passage_counts.index:
        if idx == "Missing":
            display_labels.append("Missing")
        else:
            display_labels.append(str(idx))
    
    bars = ax.barh(range(len(passage_counts)), passage_counts.values, color=colors,
                   edgecolor='black', linewidth=1.5)
    # Add hatch pattern to distinguish from location plot
    for bar in bars:
        bar.set_hatch('...')
    
    ax.set_yticks(range(len(passage_counts)))
    ax.set_yticklabels(display_labels, fontsize=10)
    ax.set_xlabel("Number of Isolates", fontsize=13)
    
    # Title with missing info
    title = f"Passage Distribution (Top {len(passage_counts)} out of {unique_passages} available"
    if missing_count > 0:
        title += f", {missing_count:,} missing"
    title += ")"
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.grid(axis='y', visible=False)
    
    # Add value labels on bars
    for i, v in enumerate(passage_counts.values):
        ax.text(v + max(passage_counts.values) * 0.01, i, f'{v:,}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "passage_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_host_subtype_heatmap_with_numbers(
    df: pd.DataFrame,
    output_dir: Path,
    top_hosts: int = 15,
    top_subtypes: int = 15) -> None:
    """
    Plot host × H/N subtype heatmap with actual numbers displayed.
    
    Same as above but shows count values in each cell.
    """
    # Filter to rows with both host and subtype
    df_plot = df[df["host"].notna() & df["hn_subtype"].notna()].copy()
    
    # Get top hosts and subtypes
    top_host_list = df_plot["host"].value_counts().head(top_hosts).index.tolist()
    top_subtype_list = df_plot["hn_subtype"].value_counts().head(top_subtypes).index.tolist()
    
    # Create crosstab
    crosstab = pd.crosstab(
        df_plot[df_plot["host"].isin(top_host_list)]["host"],
        df_plot[df_plot["hn_subtype"].isin(top_subtype_list)]["hn_subtype"]
    )
    
    # Reorder by counts for better visualization
    crosstab = crosstab.loc[top_host_list, top_subtype_list]
    
    # Create heatmap with annotations
    fig, ax = plt.subplots(figsize=(max(12, top_subtypes * 0.6), max(10, top_hosts * 0.5)))
    sns.heatmap(
        crosstab,
        annot=True,  # Show numbers
        fmt='d',
        cmap='YlOrRd',
        cbar_kws={'label': 'Number of Isolates'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={'size': 10}
    )
    ax.set_xlabel("H/N Subtype", fontsize=13)
    ax.set_ylabel("Host", fontsize=13)
    ax.set_title(f"Host × H/N Subtype Heatmap with Counts (Top {top_hosts} hosts × Top {top_subtypes} subtypes)", 
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    output_path = output_dir / "host_subtype_heatmap_with_numbers.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_year_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    start_year: int = 2000,
    show_bar_counts: bool = True) -> None:
    """
    Plot year distribution as histogram.
    
    Shows the overall temporal coverage of the dataset.
    Note: This is different from temporal trends (plot 1) - this shows
    the distribution of years, not how subtypes/hosts change over time.
    
    Args:
        show_bar_counts: If True, display count on top of each bar
    """
    df_plot = df[df["year"].notna()].copy()
    
    # Filter to start from year 2000
    df_plot = df_plot[df_plot["year"] >= start_year].copy()
    
    if len(df_plot) == 0:
        print("No year data available after filtering")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bins aligned to integer year boundaries
    year_min = int(df_plot["year"].min())
    year_max = int(df_plot["year"].max())
    bin_edges = np.arange(year_min, year_max + 2, 1)
    
    # Create histogram with aligned bins
    n, bins, patches = ax.hist(df_plot["year"], bins=bin_edges, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Set x-ticks at reasonable intervals, centered in bars
    year_range = year_max - year_min
    if year_range <= 5:
        tick_interval = 1
    elif year_range <= 15:
        tick_interval = 2
    elif year_range <= 30:
        tick_interval = 5
    else:
        tick_interval = 10
    
    # Calculate bin centers for count labels
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    tick_start = (year_min // tick_interval) * tick_interval
    tick_end = ((year_max // tick_interval) + 1) * tick_interval
    x_ticks_years = np.arange(tick_start, tick_end + 1, tick_interval)
    x_ticks_years = x_ticks_years[(x_ticks_years >= year_min) & (x_ticks_years <= year_max + 1)]
    
    # Center ticks exactly at year + 0.5 (bin centers for integer-aligned bins)
    x_ticks_centered = x_ticks_years + 0.5
    x_tick_labels = [int(year) for year in x_ticks_years]
    
    ax.set_xticks(x_ticks_centered)
    ax.set_xticklabels(x_tick_labels, rotation=90, ha='right')
    ax.set_xlim(year_min - 0.5, year_max + 1.5)
    
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Number of Isolates", fontsize=13)
    ax.set_title("Year Distribution (Overall Temporal Coverage)", fontsize=15, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Add count labels on top of bars if requested (vertical labels for every bar, positioned higher)
    if show_bar_counts:
        max_count = max(n) if len(n) > 0 else 1
        for i, (count, center) in enumerate(zip(n, bin_centers)):
            if count > 0:  # Only label bars with counts > 0
                ax.text(center, count + max_count * 0.05, f'{int(count):,}', 
                       ha='center', va='bottom', fontsize=7, rotation=90)
    
    # Add statistics text
    stats_text = f"Range: {df_plot['year'].min():.0f} - {df_plot['year'].max():.0f}\n"
    stats_text += f"Total isolates: {len(df_plot):,}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / "year_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main function to run the EDA."""
    # File paths
    data_dir = project_root / "data" / "raw" / "Full_Flu_Annos"
    key_file = data_dir / "Flu_Genomes.key"
    meta_file = data_dir / "Flu.first-seg.meta.tab"
    output_dir = project_root / "data" / "processed" / "flu" / "metadata_eda"
    
    # Create plots subdirectory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not key_file.exists():
        print(f"Error: File not found: {key_file}")
        return
    if not meta_file.exists():
        print(f"Error: File not found: {meta_file}")
        return

    # Load both files
    key_df = load_flu_genomes_key(key_file)
    meta_df = load_flu_first_seg_meta(meta_file)

    # Merge and construct analysis-ready dataframe
    df = build_analysis_dataframe(key_df, meta_df, geo_location_clean_mode=GEO_LOCATION_CLEAN_MODE_DEFAULT)

    show_cols = [
        'virus_name', 'hn_subtype', 'host_inferred', 'geo_location_clean', 'year',
        'strain', 'virus_name_meta', 'hn_subtype_meta', 'host_common_name',
        'lab_host', 'passage', 'host', 'geo_city', 'geo_state',
        'geo_country', 'geo_location_raw', 'geo_location_canonical'
    ]
    print(df[show_cols].head())

    # Perform exploratory analysis
    analyze_metadata(df)
    # jj = df[~df['geo_location_reject_reason'].isna()]

    # Check for data quality issues
    identify_data_quality_issues(df)

    # Save results
    save_summary_statistics(df, output_dir)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('Set2')
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    # 1. Temporal distributions
    plot_temporal_by_subtype(df, plots_dir, top_n=10)
    plot_temporal_by_host(df, plots_dir, top_n=10)
    plot_temporal_combined(df, plots_dir, top_subtypes=5, top_hosts=5)

    # 2. Host distribution
    plot_host_distribution(df, plots_dir, top_n=20)

    # 3. Subtype distribution (use same top_n as host for consistency)
    plot_subtype_distribution(df, plots_dir, top_n=20)

    # 4. Host × Subtype heatmap (with numbers only - removed the one without numbers)
    plot_host_subtype_heatmap_with_numbers(df, plots_dir, top_hosts=15, top_subtypes=15)

    # 5. Year distribution (with bar counts)
    plot_year_distribution(df, plots_dir, start_year=1990, show_bar_counts=True)

    # 6. Location distribution
    plot_location_distribution(df, plots_dir, top_n=20)

    # 7. Passage distribution
    plot_passage_distribution(df, plots_dir, top_n=20)

    print("\n" + "="*80)
    print("EDA COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review the parsed metadata in: {output_dir / 'flu_genomes_metadata_parsed.csv'}")
    print(f"2. Review plots in: {plots_dir}")
    print("3. Establish mapping between 'hash_id' and 'assembly_id' used in the pipeline")
    print("4. Integrate metadata with main dataset for stratified splitting")


if __name__ == '__main__':
    main()
