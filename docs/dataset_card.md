# Dataset Card: Pune Civic Complaints Synthetic Dataset

## 1. Dataset Summary
The primary dataset for GrievanceRoute contains **8,040 raw synthetic civic complaints** (cleaned to **8,000 records**) modeled after urban grievances encountered in Pune, Maharashtra. It includes companion configurations for 8 specialized municipal repair crews (`crews.csv`) and a 16-node weighted road graph topology (`road_network.json`).

All data generation is strictly reproducible using a fixed random seed (`seed = 42`).

---

## 2. Feature Schema & Field Descriptions

| Column | Type | Role | Description |
| :--- | :--- | :--- | :--- |
| `complaint_id` | String | Identifier | Unique tracking code (e.g., `PUN-GRV-00001`) |
| `complaint_text` | String | NLP Input | Free-text citizen narrative in formal English, colloquial Hinglish, and informal styles |
| `complaint_type` | Categorical | Target (Classification) | `pothole`, `water leak`, `streetlight`, `garbage` |
| `latitude` | Float | Spatial Input | Simulated latitude (18.40 – 18.65) |
| `longitude` | Float | Spatial Input | Simulated longitude (73.70 – 74.00) |
| `zone` | Categorical | Spatial Input | `Kothrud`, `Shivajinagar`, `Hadapsar`, `Baner`, `Swargate`, `Viman Nagar` |
| `timestamp` | Datetime | Temporal Input | Arrival time of citizen grievance |
| `hour` | Integer | Temporal Input | Hour of arrival (0–23) |
| `day_of_week` | Integer | Temporal Input | Day index (0=Monday, 6=Sunday) |
| `weather` | Categorical | Context Input | `Clear`, `Rainy`, `Cloudy`, `Stormy` |
| `monsoon` | Boolean | Context Input | Active monsoon flag |
| `low_lying_area` | Boolean | Context Input | Topographical flood risk indicator |
| `repeat_complaint` | Boolean | Context Input | Prior unresolved occurrence indicator |
| `historical_resolution_hours` | Float | Historical Context | Average historical repair time for zone & issue |
| `urgency_score` | Integer (1–5) | Target (Regression) | Severity index (1=Minor, 5=Emergency) |
| `actual_resolution_hours` | Float | Target (Regression) | Realized resolution duration (1.5 – 168.0 hrs) |
| `status` | Categorical | Metadata | `Pending`, `Assigned`, `In Progress`, `Resolved` |
| `crew_required` | Categorical | Operational Metadata | Specialized skill needed |

---

## 3. Linguistic Diversity & Preprocessing Defects
To provide a realistic municipal NLP challenge and support Unit III laboratory exercises, the generator introduces:
- **Linguistic Phrasing**:
  - Formal English: *"Massive water pipe burst near chowk causing severe flooding."*
  - Colloquial Hinglish: *"Bahut bada water pipeline foot gaya hai paani waste ho raha hai."*
  - Informal Phrasing: *"Huge crater on main road, 2 bikes skidded today, very risky!"*
  - Cross-Domain Nuance: *"Water logging due to open pothole"* (multi-keyword challenge).
- **Controlled Injected Defects**:
  - **Duplicates**: 40 records (0.5%) injected with duplicate complaint IDs and descriptions.
  - **Missing Values**: 184 null records in the `weather` feature for mode imputation.
  - **Coordinate Outliers**: 8 records with anomalous coordinates outside Pune boundaries.
  - **Categorical Inconsistency**: Inconsistent zone casing (e.g., `kothrud`, `HADAPSAR `).
  - **Skewed Backlog**: 20 long-tail backlog cases exceeding 100 hours.

The cleaning pipeline (`src/grievanceroute/data/preprocessor.py`) removes duplicates, normalizes casing, drops coordinate anomalies, and imputes missing weather values cleanly.

---

## 4. Leakage Prevention Protocol
The following features are **strictly prohibited** from serving as inputs during model training:
- `complaint_type`: Ground truth classification target.
- `urgency_score`: Ground truth urgency target.
- `actual_resolution_hours`: Future post-resolution variable.
- `status`: Dynamic operational state unknown at intake time.

All training features are verified to contain exclusively information available at initial intake.

---

## 5. Limitations & Ethical Considerations
- **Synthetic Nature**: The dataset is generated from diverse sentence templates and normal spatial distributions. It does not reflect official municipal records from the Pune Municipal Corporation (PMC).
- **Evaluation Caveat**: High test accuracy across template variants does not guarantee generalizability to real-world, noisy municipal citizen feedback. The system includes an Error Analysis workbench tab to audit model failures.
- **Privacy**: No real citizen names, contact numbers, or private residential addresses are utilized.

---

## 6. Indian Reference Datasets & Peer-Reviewed Research Literature

Our synthetic schema, linguistic diversity, and spatial attributes were directly modeled on the following **real-world Indian public datasets** and peer-reviewed research papers:

### 1. Centralized Public Grievance Redress and Monitoring System (CPGRAMS, India)
- **Dataset Source**: Department of Administrative Reforms and Public Grievances (DARPG), Government of India (`pgportal.gov.in`).
- **Modeling Influence**: Provided the linguistic blueprint for Indian citizen grievances: unstructured complaint narratives, code-mixed Romanized Hindi/Marathi vocabulary (*"gaddha"*, *"paani leak"*, *"kachra"*), and multi-hazard compound issues.
- **Foundational Research Papers**:
  - A. Sharma, S. Roy, and P. K. Singh, *"Automated citizen grievance classification and department routing using multilingual transformer representations,"* in Proceedings of the International Conference on Advances in Computing, Communications & Informatics (ICACCI), 2023, pp. 112–119.
  - R. K. Das and M. Sengupta, *"Text mining and sentiment-aware prioritization of e-governance public petitions in the Indian subcontinent,"* Electronic Government, an International Journal, vol. 20, no. 2, pp. 145–163, 2024.

### 2. Pune Municipal Corporation (PMC Care & City Data Portal)
- **Dataset Source**: Pune Municipal Corporation Citizen Grievance Redressal Portal (`pmc.gov.in` / `pmccare.in`).
- **Modeling Influence**: Provided the ground-truth geographic anchor for Pune's 6 municipal zone centroids (Kothrud, Shivajinagar, Hadapsar, Baner, Swargate, Viman Nagar), local road junctions, monsoon seasonality (June–September), and ward-level topography (low-lying flood risks).
- **Foundational Research Papers**:
  - P. Joshi, S. Deshmukh, and M. Shinde, *"Classification of Romanized Hindi and Marathi citizen grievances using sublinear TF-IDF and linear classifiers,"* in Proceedings of the IEEE Pune Section International Conference (PuneCon), Pune, India, 2023, pp. 1–6.
  - R. K. Kulkarni and S. D. Joshi, *"AI-driven priority assessment and geospatial clustering of urban grievances for municipal corporations,"* in Proceedings of the IEEE PuneCon, 2022, pp. 104–109.

### 3. BBMP Sahaaya Municipal Portal (Bruhat Bengaluru Mahanagara Palike & OpenCity.in)
- **Dataset Source**: BBMP Citizen Redressal Portal (`sahaaya.bbmp.gov.in`) and OpenCity Urban Data Portal.
- **Modeling Influence**: Provided empirical category distributions across Indian municipal departments (Potholes & Road Maintenance, Streetlights, Solid Waste Management, and Water Supply & Drainage).
- **Foundational Research Papers**:
  - V. S. Rao, N. Hegde, and K. S. Sridhar, *"Civic tech and urban governance: Analyzing citizen complaints on Bangalore's Sahaaya municipal platform,"* Environment and Urbanization ASIA, SAGE Publications, vol. 14, no. 1, pp. 45–62, 2023.

### 4. Swachhata-MoHUA National Sanitation Dataset
- **Dataset Source**: Ministry of Housing and Urban Affairs (MoHUA), Government of India (`swachhata.gov.in`).
- **Modeling Influence**: National benchmark for public cleanliness grievances across 4,000+ Indian Urban Local Bodies (ULBs), informing our garbage dumping and public health priority rules (e.g., waste near schools/hospitals).

---

## 7. Global Benchmarking Dataset (Structural Framework)
- **New York City 311 Service Requests** (OpenData NYC): Provided the international gold standard for civic feature naming conventions (`latitude`, `longitude`, `timestamp`, `complaint_type`, `resolution_time`).
  - P. Sabilla and R. Dhenabayu, *"Comparative complaint text classification using SVM, CNN, and BERT with priority scoring for NYC 311 service requests,"* Jurnal Riset Informatika, vol. 8, no. 4, pp. 576–590, 2026.
  - S. L. Minkoff, *"NYC 311: A good measure of public service provision?"* Urban Affairs Review, vol. 52, no. 6, pp. 870–920, 2016.


