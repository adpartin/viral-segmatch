# Preprocessing Stage

The first stage of the viral-segmatch pipeline, converting raw GTO files into clean protein sequences.

## 🎯 Purpose

Transform raw viral protein data from GTO files into standardized protein sequences suitable for ESM-2 embedding generation.

## 📊 Input/Output

### Input
- **Raw GTO files**: Original viral protein data
- **Location**: `data/raw/`
- **Format**: GTO (GenBank Table Output) files

### Output
- **Protein sequences**: Clean, standardized protein data
- **Location**: `data/processed/{virus}/{data_version}/`
- **Format**: `protein_final.csv`

## 🔧 Scripts

### Flu A Preprocessing
```bash
./scripts/preprocess_flu_protein.sh
```

### Bunyavirus Preprocessing
```bash
./scripts/preprocess_bunya_protein.sh
```

## 📋 Process Steps

### 1. Data Loading
- Load GTO files from `data/raw/`
- Parse protein sequences and metadata
- Extract relevant protein information

### 2. Filtering
- Filter by selected protein functions
- Remove low-quality sequences
- Apply length and quality filters

### 3. Standardization
- Standardize sequence formats
- Clean protein names and identifiers
- Ensure consistent data structure

### 4. Output Generation
- Save to `protein_final.csv`
- Include metadata (function, length, quality)
- Generate processing summary

## 📁 Output Structure

```
data/processed/
└── {virus}/
    └── {data_version}/
        ├── protein_final.csv          # Main output
        ├── processing_summary.txt     # Processing stats
        └── quality_report.txt         # Data quality metrics
```

## 🔍 Key Parameters

### Data Sampling
```yaml
# Optional: Limit number of files processed
max_files_to_process: null  # Process all files
# max_files_to_process: 100  # Process only 100 files
```

### Protein Selection
```yaml
virus:
  selected_functions: [pb1, pb2, pa]  # Flu A proteins
  # selected_functions: [l, m, s]     # Bunyavirus proteins
```

### Quality Filters
- **Minimum length**: Remove sequences below threshold
- **Maximum length**: Remove sequences above threshold
- **Quality score**: Remove low-quality sequences
- **Function filter**: Only include selected protein functions

## 📊 Output Format

### `protein_final.csv` Columns
- **`assembly_id`**: Unique identifier for each protein
- **`protein_name`**: Protein name/identifier
- **`function`**: Protein function (pb1, pb2, pa, etc.)
- **`sequence`**: Protein amino acid sequence
- **`length`**: Sequence length
- **`quality_score`**: Data quality metric
- **`source_file`**: Original GTO file

### Example Data
```csv
assembly_id,protein_name,function,sequence,length,quality_score,source_file
FLU001,PB1,pb1,MAKLLVLLFATAG,13,0.95,flu_001.gto
FLU002,PB2,pb2,MSLLTEVETPIRNEWG,16,0.92,flu_001.gto
FLU003,PA,pa,MAKLLVLLFATAG,13,0.98,flu_001.gto
```

## 🔧 Customization

### Adding New Viruses
1. **Create preprocessing script**:
   ```bash
   # Copy existing script
   cp scripts/preprocess_flu_protein.sh scripts/preprocess_my_virus.sh
   ```

2. **Modify script parameters**:
   ```bash
   # Update virus-specific settings
   VIRUS_NAME="my_virus"
   DATA_VERSION="January_2025"
   SELECTED_FUNCTIONS="protein1,protein2,protein3"
   ```

3. **Update configuration**:
   ```yaml
   # Add to conf/bundles/my_virus.yaml
   virus:
     virus_name: my_virus
     data_version: January_2025
     selected_functions: [protein1, protein2, protein3]
   ```

### Custom Quality Filters
```python
# In preprocessing script
def apply_quality_filters(df):
    # Length filter
    df = df[df['length'] >= MIN_LENGTH]
    df = df[df['length'] <= MAX_LENGTH]
    
    # Quality score filter
    df = df[df['quality_score'] >= MIN_QUALITY]
    
    # Function filter
    df = df[df['function'].isin(SELECTED_FUNCTIONS)]
    
    return df
```

## 📈 Quality Metrics

### Processing Summary
- **Total sequences**: Number of sequences processed
- **Filtered sequences**: Sequences after quality filtering
- **Function distribution**: Count by protein function
- **Length statistics**: Mean, median, range of sequence lengths

### Quality Report
- **Quality score distribution**: Histogram of quality scores
- **Length distribution**: Histogram of sequence lengths
- **Function coverage**: Sequences per protein function
- **Data completeness**: Missing data analysis

## 🔍 Troubleshooting

### Common Issues

**1. No data found**
```bash
# Check input directory
ls -la data/raw/

# Verify file format
file data/raw/*.gto
```

**2. Empty output**
```bash
# Check quality filters
grep "quality_score" data/processed/flu/July_2025/protein_final.csv | head

# Check function filter
cut -d',' -f3 data/processed/flu/July_2025/protein_final.csv | sort | uniq -c
```

**3. Memory issues**
```bash
# Process files in batches
max_files_to_process: 100  # Instead of processing all at once
```

### Debugging Tips

**1. Check intermediate files**
```bash
# Look for temporary files
ls -la data/processed/flu/July_2025/
```

**2. Verify data quality**
```bash
# Check sequence lengths
cut -d',' -f5 data/processed/flu/July_2025/protein_final.csv | sort -n

# Check quality scores
cut -d',' -f6 data/processed/flu/July_2025/protein_final.csv | sort -n
```

**3. Test with small dataset**
```bash
# Process only a few files for testing
max_files_to_process: 10
```

## 📝 Best Practices

### Data Quality
- **Validate input**: Check GTO file format and content
- **Apply filters**: Remove low-quality sequences
- **Document issues**: Log any data quality problems

### Performance
- **Batch processing**: Process large datasets in batches
- **Memory management**: Monitor memory usage
- **Progress tracking**: Log processing progress

### Reproducibility
- **Seed setting**: Use consistent random seeds
- **Version control**: Track data versions
- **Documentation**: Record processing parameters
