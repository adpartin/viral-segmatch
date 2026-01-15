# Geographic Location Extraction and Standardization Analysis

## Executive Summary

**Current Status (UPDATED):**
- 92.5% of records have `geo_location_clean` (103,414/111,797)
- **Column structure (IMPLEMENTED):** `geo_location_raw`, `geo_location_clean`, `geo_city`, `geo_state`, `geo_country`
- **Fixes implemented:**
  - ✅ Fixed parsing bug: "A/Cattle/USA/..." now correctly extracts "USA" as location
  - ✅ Renamed columns: `city` → `geo_city`, `state` → `geo_state`, `country` → `geo_country`
  - ✅ Removed redundant columns: `location`, `geo_location`
  - ✅ Added Chinese province mappings (12 provinces → geo_state with country="China")
  - ✅ Added country-only location mappings (19 countries)
  - ✅ Added geographic feature mappings (Interior Alaska, Delaware Bay, etc.)
  - ✅ Added special region mappings (Hong Kong → China)
  - ✅ Added "cattle", "cow" to host keywords and non-location terms

---

## Process Flow

### 1. Initial Extraction (`parse_virus_name()`)

**Input:** Virus name string (e.g., `"Influenza A virus A/Rhode Island/62/2023"`)

**Process:**
1. Split by `/` and extract parts after "Influenza A virus A"
2. Determine if first part is host or location:
   - Check against `host_keywords` list (140+ terms: chicken, swine, duck, etc.)
   - Check against `non_location_keywords` (environment, unknown, na, human)
   - Use word boundary matching to avoid false positives
3. Extract location:
   - **If host present:** `parts[1]` is location (unless it's a strain identifier)
   - **If no host:** `parts[0]` is location
4. Strain detection: If `parts[1]` matches pattern `\d+[-_]\d+` with 3+ digits, treat as strain, not location

**Output:** Inferred location token from the name (a single string), stored in the key table as:
- `geo_location_inferred` (e.g., `"Rhode Island"`, `"PA"`, `"Hunan"`)

**Known Issues:**
- ✅ **FIXED:** "A/Cattle/USA/..." format now correctly extracts "USA" as location. Logic checks if `parts[1]` is a host term and `parts[2]` exists, then uses `parts[2]` as location.
- Only extracts first location part; if both state and country appear, only state is captured (by design)

### 2. Standardization (`standardize_location()`)

**Input:** Raw location string from step 1

**Process (sequential checks):**

1. **Filtering (early exit if matches):**
   - Numeric-only values (years, parsing errors)
   - Strain identifiers (patterns like "23-038138-001-original")
   - Alphanumeric codes (6+ chars, no spaces, mix of letters/numbers)
   - Non-location terms (440+ terms: bird species, animals, scientific names)
   - Lowercase-starting strings (parsing errors)
   - Short codes (≤2 chars, not valid US state abbreviations)

2. **Extraction from parentheses:**
   - Pattern: `"X-53A(Puerto Rico"` → `"Puerto Rico"`

3. **Normalization:**
   - Whitespace normalization (multiple spaces → single)
   - Hyphen normalization (en-dash, em-dash → standard hyphen)
   - Name standardization: `"Ad Dawhah"` → `"Ad-Dawhah"`, `"Aguascallientes"` → `"Aguascalientes"`, `"DC"` → `"District of Columbia"`

4. **Mapping (priority order):**
   - **US state abbreviations:** `"PA"` → state=`"Pennsylvania"`, country=`"USA"`
   - **Cities:** `"Chicago"` → city=`"Chicago"`, state=`"Illinois"`, country=`"USA"`; `"Lyon"` → city=`"Lyon"`, country=`"France"`
   - **Province abbreviations:** `"ALB"` → state=`"Alberta"`, country=`"Canada"`
   - **Provinces/states:** `"Alberta"` → state=`"Alberta"`, country=`"Canada"`
   - **German states:** `"Germany-HE"` → state=`"Germany-HE"`, country=`"Germany"`
   - **US states (full name):** `"Rhode Island"` → state=`"Rhode Island"`, country=`"USA"`
   - **Country standardization:** `"United States"` → country=`"USA"`

5. **Default:** If no mapping found, return `geo_location_clean` = input string, no city/state/country

**Output:** Dictionary with:
- `geo_location_clean`: Standardized location name (for filtering)
- `city`: City name (if applicable, 4.7% coverage) - stored as `geo_city` in DataFrame
- `state`: State/province name (if applicable, 60.1% coverage) - stored as `geo_state` in DataFrame
- `country`: Country name (if applicable, 62.5% coverage) - stored as `geo_country` in DataFrame

### 3. DataFrame Construction (`build_analysis_dataframe()`)

**Process:**
1. Merge `Flu_Genomes.key` with `Flu.first-seg.meta.tab` on `hash_id`
2. Apply `standardize_location()` to each raw location value
3. Create columns (important distinction):
   - `geo_location_raw`: the raw token preserved from `geo_location_inferred` (i.e., “what we extracted from the name”)
   - `geo_location_canonical`: the canonical standardized token produced by `standardize_location()` (historical meaning of `geo_location_clean`)
   - `geo_city`, `geo_state`, `geo_country`: structured fields inferred by `standardize_location()`
   - `geo_location_clean`: the *analysis* “clean” value, chosen by a policy:
     - Default policy: **USA → `geo_state` if available; otherwise `geo_country`**.
     - Other modes are possible (city/state/country/canonical).
   - **Removed:** `geo_location_inferred` column after `geo_location_raw` is created (to avoid confusion with genomic `location`)

**Current Column Structure (IMPLEMENTED):**
```
geo_location_raw       # Raw extracted location token from virus name (preserved for reference)
geo_location_canonical # Canonical standardized token from standardize_location()
geo_location_clean     # Analysis "clean" location (policy-driven; default: US->state else country)
geo_city           # City name (4.7% populated)
geo_state          # State/province (60.1% populated)
geo_country        # Country (62.5% populated)
```

**Note:** The `geo_` prefix distinguishes geographic locations from the genomic `location` column in GTO features (which contains genomic coordinates like `[["NC_086346.1", "70", "+", "738"]]`).

---

## Data Quality Issues

### Critical Issues

1. **"Cattle" as Location (2,071 entries)** ✅ **FIXED**
   - **Root cause:** Virus names like `"A/Cattle/USA/24-021715-005/2024"` were parsed incorrectly
   - **Previous behavior:** `parts[0]`="Cattle" (host), `parts[1]`="USA" (should be location), but parser set location=`parts[1]`="Cattle"
   - **Fix implemented:** 
     - Added "cattle" to `host_keywords` list
     - Updated logic: when host is detected and `parts[1]` is also a host term, check if `parts[2]` exists and use it as location
     - Added "cattle", "cow" to `NON_LOCATION_TERMS` as backup filtering
   - **Result:** "A/Cattle/USA/24-021715-005/2024" now correctly extracts host="Cattle", location="USA"

2. **Missing State/Country Mappings (33,543 entries)** ✅ **PARTIALLY FIXED**
   - ✅ **Fixed:** Chinese provinces (12): Added to `province_to_country` mapping → `geo_state` with `geo_country="China"`
   - ✅ **Fixed:** Geographic features: "Interior Alaska" → `geo_state="Alaska"`, `geo_country="USA"`; "Delaware Bay" → `geo_state="Delaware"`, `geo_country="USA"`
   - ✅ **Fixed:** Country-only locations (19): Added `country_only_locations` mapping (Netherlands, Vietnam, Italy, etc.)
   - ✅ **Fixed:** Special regions: "Hong Kong" → `geo_state="Hong Kong"`, `geo_country="China"`
   - **Remaining:** Some locations may still lack mappings (will be addressed in metadata regeneration)

3. **Inconsistent Column Naming** ✅ **FIXED**
   - ✅ Removed redundant `geo_location` column (was duplicate of `geo_location_clean`)
   - ✅ Removed redundant `location` column (was duplicate of `geo_location_raw`, and conflicts with genomic 'location')
   - ✅ Renamed: `city` → `geo_city`, `state` → `geo_state`, `country` → `geo_country`
   - ✅ Final structure: `geo_location_raw`, `geo_location_clean`, `geo_city`, `geo_state`, `geo_country`

### Moderate Issues

4. **Case Variations**
   - "CATTLE" (91 entries) vs "Cattle" (2,071) - should be filtered, not standardized
   - Some locations have inconsistent casing

5. **City Mapping Inconsistencies**
   - Some cities mapped to countries (e.g., "Sydney" → country="Australia") but `geo_location_clean` = "Sydney" (should be "Australia" or state)
   - Only 4.7% of records have city populated

6. **Country-Only Locations**
   - 2,675 entries have country but no state (e.g., "England" → country="United Kingdom", but no state)
   - Some are cities that should map to states/countries but don't

---

## Current Coverage Statistics

**From 111,797 total records:**
- `geo_location_clean`: 103,414 (92.5%)
- `state`: 67,196 (60.1%)
- `country`: 69,871 (62.5%)
- `city`: 5,222 (4.7%)

**Breakdown:**
- Records with city+state+country: 3,098 (2.8%)
- Records with state+country (no city): 64,098 (57.3%)
- Records with only country: 551 (0.5%)
- Records with only `geo_location_clean` (no state/country): 33,543 (30.0%)

---

## Top Locations (Issues Highlighted)

1. **Michigan** (8,140) - ✅ Correct (state+country)
2. **California** (5,021) - ✅ Correct
3. **New York** (4,412) - ✅ Correct
4. **USA** (3,236) - ⚠️ Country-level only, no state
5. **Minnesota** (2,709) - ✅ Correct
6. **Cattle** (2,071) - ✅ **FIXED: Now correctly parsed as host, location extracted from next part**
7. **Massachusetts** (2,683) - ✅ Correct
8. **Iowa** (2,646) - ✅ Correct
9. **Texas** (2,582) - ✅ Correct
10. **Ohio** (2,398) - ✅ Correct
11. **Wisconsin** (2,206) - ✅ Correct
12. **Washington** (2,055) - ✅ Correct
13. **Jiangxi** (1,066) - ✅ **FIXED: Now maps to geo_state="Jiangxi", geo_country="China"**
14. **Netherlands** (964) - ✅ **FIXED: Now maps to geo_country="Netherlands"**
15. **Hong Kong** (887) - ✅ **FIXED: Now maps to geo_state="Hong Kong", geo_country="China"**
16. **Interior Alaska** (714) - ✅ **FIXED: Now maps to geo_state="Alaska", geo_country="USA"**
17. **Delaware Bay** (570) - ✅ **FIXED: Now maps to geo_state="Delaware", geo_country="USA"**

---

## Recommendations

### 1. Fix Parsing Bug (Priority: HIGH) ✅ **IMPLEMENTED**

**Problem:** "A/Cattle/USA/..." extracted "Cattle" as location

**Solution Implemented:** 
- Added "cattle" to `host_keywords` list
- Updated logic in `parse_virus_name()`: when host is detected and `parts[1]` is also a host term, check if `parts[2]` exists and use it as location
- Added "cattle", "cow" to `NON_LOCATION_TERMS` as backup filtering

**Result:** "A/Cattle/USA/24-021715-005/2024" now correctly extracts:
- host="Cattle"
- location="USA" 
- strain="24-021715-005"
- year="2024"

### 2. Rename Columns (Priority: HIGH) ✅ **IMPLEMENTED**

**Changes Made:**
- ✅ `location` → **REMOVED** (redundant with `geo_location_raw`, conflicts with genomic 'location')
- ✅ `geo_location_raw` → **KEPT** (raw extracted location)
- ✅ `geo_location_clean` → **KEPT** (standardized, used for filtering)
- ✅ `geo_location` → **REMOVED** (duplicate of `geo_location_clean`)
- ✅ `city` → `geo_city`
- ✅ `state` → `geo_state`
- ✅ `country` → `geo_country`

**Rationale:**
- `geo_` prefix clarifies these are geographic (not genomic) locations
- Removes redundancy
- Consistent naming convention
- Avoids conflict with genomic 'location' column in protein DataFrames

### 3. Expand Country/State Mappings (Priority: MEDIUM) ✅ **IMPLEMENTED**

**Mappings Added:**
- ✅ Chinese provinces (12): Added to `province_to_country` → `geo_state` with `geo_country="China"`
  - Rationale: Chinese provinces are first-level administrative divisions (like US states), so mapping to `geo_state` is appropriate
- ✅ Geographic features: Added `geographic_features` mapping
  - "Interior Alaska" → `geo_state="Alaska"`, `geo_country="USA"`
  - "Delaware Bay" → `geo_state="Delaware"`, `geo_country="USA"`
- ✅ Special regions: Added `special_regions` mapping
  - "Hong Kong" → `geo_state="Hong Kong"`, `geo_country="China"`
- ✅ Country-only locations (19): Added `country_only_locations` mapping
  - Ensures country-level locations have `geo_country` populated

### 4. Filter "Cattle" and Similar Host Terms (Priority: HIGH) ✅ **IMPLEMENTED**

**Changes Made:**
- ✅ Added "cattle" to `host_keywords` list (primary fix at parsing level)
- ✅ Added "cattle", "cow" to `NON_LOCATION_TERMS` in `standardize_location()` (backup filtering)

### 5. Improve City-to-State/Country Mapping (Priority: LOW)

- Current: Only 4.7% have city populated
- Most cities are already mapped correctly when present
- Low priority unless city-level filtering is needed

---

## Usage in Dataset Filtering

**Current implementation (`dataset_segment_pairs.py`):**
- Filters on `geo_location_clean` using `.isin([filter_value])`
- Exact match required (case-sensitive)
- Filter value comes from config: `config.dataset.geo_location`

**Example:**
```yaml
dataset:
  geo_location: "Jiangxi"  # Must match geo_location_clean exactly
```

**Considerations (UPDATED):**
- `geo_location_clean` is now designed to be the primary downstream filtering column, but its meaning depends on the chosen policy.
- Default policy is intended for stratification/control: **US → state, non-US → country**.
- If you need the canonical token (legacy behavior), filter on `geo_location_canonical`.

---

## Implementation Status

### Phase 1: Critical Fixes ✅ **COMPLETED**
1. ✅ Fixed "Cattle" parsing bug in `parse_virus_name()`
2. ✅ Added "cattle"/"cow" to `NON_LOCATION_TERMS` as backup
3. ✅ Renamed columns: `city` → `geo_city`, `state` → `geo_state`, `country` → `geo_country`
4. ✅ Removed redundant columns: `location`, `geo_location`

### Phase 2: Data Quality ✅ **COMPLETED**
5. ✅ Added Chinese province → country mappings (12 provinces)
6. ✅ Added geographic feature mappings (Interior Alaska, Delaware Bay, Bay of Plenty)
7. ✅ Added country-only location mappings (19 countries)
8. ✅ Added special region mappings (Hong Kong)

### Phase 3: Validation ⏳ **PENDING**
8. ⏳ Regenerate metadata file: `python src/preprocess/flu_genomes_eda.py`
9. ⏳ Verify "Cattle" no longer appears in `geo_location_clean`
10. ⏳ Verify coverage statistics improve (especially `geo_country` coverage)
11. ⏳ Test filtering with updated column names

---

## Additional Issues Found

### Location Categories Without State/Country Mapping

**1,451 unique locations lack state/country mapping. Breakdown:**

- **Chinese provinces (12):** Jiangxi, Guangdong, Anhui, Fujian, Zhejiang, Hunan, Shantou, Guangxi, Shandong, Dongguan, Henan, Jiangsu
  - **Fix:** Add to province mapping: `"Jiangxi": ("Jiangxi", "China")`, etc.

- **Countries (19):** USA, Netherlands, Vietnam, Italy, Singapore, Bangladesh, Nicaragua, China, Peru, Sweden, France, Egypt, Denmark, South Korea, Korea, Taiwan, Thailand, Mexico, Chile
  - **Fix:** Add country-level entries: `"Netherlands": (None, "Netherlands")`, etc.

- **Geographic features (5):** Interior Alaska, Delaware Bay, Bayern, Baylor, Bay of Plenty
  - **Fix:** "Interior Alaska" → state="Alaska", country="USA"; "Delaware Bay" → state="Delaware", country="USA"; "Bayern" → state="Bavaria", country="Germany"

- **Cities (2):** Hong Kong, Managua
  - **Fix:** "Hong Kong" → state="Hong Kong", country="China"; "Managua" → city="Managua", country="Nicaragua"

- **Other issues:**
  - "African Stonechat" - bird species, should be filtered
  - "AUS" - abbreviation, should map to "Australia"
  - Many international city names need country mapping

## Questions/Uncertainties

1. **"Hong Kong" handling:** Should it map to country="China" or be treated as special administrative region?
2. **"USA" as location:** Should entries with `geo_location_clean="USA"` have `geo_country="USA"` and `geo_state=None`, or should we try to infer state from other metadata?
3. **City-level filtering:** Is city-level filtering needed, or is state/country sufficient?
4. **Strain identifier detection:** Current regex may miss some patterns - should we expand?
5. **Abbreviation handling:** Should "AUS" map to "Australia", "KOR" to "South Korea", etc.? Need comprehensive abbreviation list?

---

## Files to Modify

1. `src/preprocess/flu_genomes_eda.py`
   - `parse_virus_name()`: Fix "Cattle" bug
   - `standardize_location()`: Add mappings, improve filtering
   - `build_analysis_dataframe()`: Rename columns
2. `src/utils/metadata_enrichment.py`
   - Update column references if needed
3. `src/datasets/dataset_segment_pairs.py`
   - Update column references: `geo_location_clean` (already correct)
4. `conf/dataset/default.yaml`
   - Already uses `geo_location` (should reference `geo_location_clean`)

---

## Testing Checklist

**After code implementation (✅ COMPLETED):**
- ✅ "Cattle" parsing bug fixed (tested: "A/Cattle/USA/..." correctly extracts location="USA")
- ✅ Column names updated: `geo_location_raw`, `geo_location_clean`, `geo_city`, `geo_state`, `geo_country`
- ✅ "Jiangxi" mapping added (geo_state="Jiangxi", geo_country="China")
- ✅ "Interior Alaska" mapping added (geo_state="Alaska", geo_country="USA")
- ✅ All code references updated

**After metadata regeneration (⏳ PENDING):**
- [ ] Regenerate metadata: `python src/preprocess/flu_genomes_eda.py`
- [ ] Verify "Cattle" no longer appears in `geo_location_clean` (should be 0 entries)
- [ ] Verify "Jiangxi" entries have `geo_country="China"`
- [ ] Verify "Interior Alaska" entries have `geo_state="Alaska"`, `geo_country="USA"`
- [ ] Verify filtering with `geo_location="Jiangxi"` works correctly
- [ ] Verify coverage statistics: `geo_location_clean` ≥ 92%, `geo_country` ≥ 70% (expected improvement from ~62.5%)

