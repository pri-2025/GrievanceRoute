# GrievanceRoute — Simple, Complete System Explanation

> **What is this guide?**  
> This document explains the entire GrievanceRoute project in plain, simple language: what the problem is, how the system solves it step-by-step, which Machine Learning models and Artificial Intelligence algorithms are used in each part, and which file in the code does what.

---

## 🌟 1. The Big Picture: What Problem Does This Solve?

Imagine you are the municipal commissioner of a big city like Pune. Every day, thousands of citizen complaints pour in:
- *"There is a huge pothole on Karve Road, bikes are slipping!"*
- *"A water supply pipe burst in Shivajinagar and is flooding the colony."*
- *"Dangerous live wire hanging and sparking near FC Road!"*
- *"Garbage has been rotting outside our society gate for 3 days."*

### In a traditional municipal system:
1. People have to manually read each complaint and guess which department should handle it.
2. Life-threatening emergencies (like live electric wires) get stuck in the same slow queue as normal complaints.
3. Repair crews drive haphazardly across the city, wasting fuel, getting stuck in traffic, and running into overtime.

### What GrievanceRoute does:
GrievanceRoute is an **automated, intelligent system** that acts as the city's brain:
1. **Reads the text** (even informal Hinglish like *"sadak pe gaddha hai"*) and categorizes it in milliseconds.
2. **Flags life-threatening hazards** immediately using emergency safety rules.
3. **Estimates urgency (1 to 5)** and how many hours the repair will take.
4. **Finds geographic hotspots** on the Pune map to see where problems are concentrating.
5. **Plans optimized routes for 8 repair crews** so they visit maximum locations in minimum kilometers and finish strictly within their 8-hour shift with **zero overtime**.

---

## 🔄 2. End-to-End Workflow (How Data Moves Through the System)

```text
[Citizen Files Complaint]
       │
       ▼
[Step 1: Validation & Cleaning] ── Checks GPS bounds, removes duplicates, cleans text.
       │
       ▼
[Step 2: Emergency Safety Check] ── Detects "live wire" / "sparking"? ──► FORCES Urgency 5/5
       │ (No emergency)                                                    & Electrical Crew!
       ▼
[Step 3: NLP Text Classification] ── TF-IDF + Logistic Regression identifies category
       │                            (Pothole, Water Leak, Streetlight, Garbage).
       ▼
[Step 4: Urgency & Time Prediction] ── Random Forest predicts Urgency (1-5) & Hours to fix.
       │
       ▼
[Step 5: Spatial Hotspots] ── K-Means finds cluster centers; DBSCAN finds dense corridors.
       │
       ▼
[Step 6: Route Optimization] ── A* finds road paths; 2-Opt & Simulated Annealing optimize
       │                        crew schedules under an 8-hour shift limit.
       ▼
[Step 7: Saved in SQLite DB] ── Dispatched to crews, tracked from 'Pending' to 'Resolved'.
```

---

## 🧩 3. Step-by-Step Breakdown: Which Part Does What & How It's Implemented

---

### Step 1: Data Generation, Preprocessing & Quality
* **Where the code lives:**
  - `src/grievanceroute/data/generator.py`
  - `src/grievanceroute/data/preprocessor.py`
  - `src/grievanceroute/data/feature_engineering.py`
  - `pages/07_Data_Quality.py`

#### What it does:
Before you can train AI models, you need data. Since real municipal records are often private or incomplete, the project has an intelligent data generator that created **8,040 realistic Pune complaints**.
- It uses realistic sentence templates in English, Marathi, Hindi, and Hinglish.
- It covers 6 real Pune zones: Kothrud, Shivajinagar, Hadapsar, Baner, Swargate, Viman Nagar.
- It intentionally introduces real-world defects (40 duplicate tickets, 170 missing weather fields, bad GPS coordinates) to demonstrate data cleaning.

#### How it is implemented:
1. **Deduplication:** Looks at `complaint_id` and drops repeated records.
2. **Missing Value Imputation:** If weather is blank, it fills it with the **mode** (the most frequent weather, e.g., "Clear").
3. **Coordinate Clamping:** Checks if latitude and longitude are within Pune city boundaries ($18.40^\circ \text{ to } 18.65^\circ \text{ N}$, $73.70^\circ \text{ to } 74.00^\circ \text{ E}$). If outside, it clips them to the city boundary.
4. **Planar Metric Projection (Equirectangular Projection):**
   - *The Problem:* GPS degrees (lat/lon) are on a sphere. 1 degree of longitude in Pune is not the same distance as 1 degree in London. If you feed raw lat/lon into Euclidean distance or K-Means, distance calculations become distorted.
   - *The Solution:* The code converts $(\text{lat}, \text{lon})$ into real flat metric kilometers $(x_{\text{km}}, y_{\text{km}})$ relative to Pune center $(18.5204^\circ\text{N}, 73.8567^\circ\text{E})$.
5. **No Target Leakage:** Future information (like actual resolution time or crew feedback) is strictly stripped before model training so the model only learns from information available at submission time.

---

### Step 2: Emergency Safety Overrides (Expert Rule Engine)
* **Where the code lives:**
  - `src/grievanceroute/services/prediction_service.py`

#### What it does:
Pure Machine Learning is statistical—it can be 96% or 99% accurate, but it can still make a rare mistake. But in civic operations, if someone reports a **"live electrical wire hanging and sparking near a bus stop"**, an algorithm cannot afford to guess wrongly. Human lives are at stake.

#### How it is implemented:
- Before running any ML model, the text is checked against a deterministic list of safety keywords:
  - `"live wire"`, `"exposed wire"`, `"sparking"`, `"transformer blast"`, `"electrocution"`.
- If any keyword matches:
  - Category is **automatically forced** to `streetlight` (Electrical Department).
  - Urgency is **forced to the absolute maximum: 5 / 5**.
  - A red warning banner is generated for supervisors.
- *AI Concept:* This is a classical **Production Rule / Expert System** (Unit I of AI syllabus).

---

### Step 3: NLP Complaint Classification (Supervised Machine Learning)
* **Where the code lives:**
  - `src/grievanceroute/ml/classifier.py`
  - `pages/04_ML_Workbench.py`

#### What it does:
When a citizen writes a complaint in plain words, this model reads the text and predicts which of the 4 departments must handle it:
1. `pothole` (Road Maintenance)
2. `water leak` (Water Supply Crew)
3. `streetlight` (Electrical Crew)
4. `garbage` (Sanitation Crew)

#### Which Model & NLP Technique is Used?
1. **Feature Extraction — TF-IDF (Term Frequency - Inverse Document Frequency):**
   - Computers cannot read words; they only understand numbers.
   - TF-IDF turns words into numbers based on how important they are:
     - If a word appears often in one complaint (e.g., *"pipe"*, *"burst"*), its weight increases.
     - If a word appears in almost all complaints (like *"the"*, *"is"*, *"pune"*), its weight decreases.
   - The code uses **Unigrams + Bigrams** (`ngram_range=(1, 2)`), meaning it looks at single words (*"wire"*, *"leak"*) and two-word pairs (*"live wire"*, *"water leak"*).
2. **Classifier — Multinomial Logistic Regression (Primary Model):**
   - Calculates the probability of each category and picks the highest one.
   - *Why Logistic Regression?* For text classification with high-dimensional sparse TF-IDF vectors, Logistic Regression is extremely fast (trains in 0.34s), doesn't overfit, and gives clear probability scores.
3. **Comparison Against Baselines:**
   - **Majority Class Baseline (`DummyClassifier`):** Always guesses the most frequent class. Accuracy = **26.87%**.
   - **Our Logistic Regression:** Accuracy = **96.37%**, Macro F1 = **0.9635**.
   - Also compared with **Multinomial Naive Bayes** (96.37%) and **Decision Tree** (96.25%).
4. **Error Analysis (Why is it not 100%?):**
   - The remaining ~3.6% errors are intentional and realistic! For example, if a citizen writes: *"Water pipeline burst flooded the street and created a massive pothole"*, the text mentions both water and pothole. The Error Analysis tab in the GUI lets examiners inspect these real-world boundary cases.

---

### Step 4: Urgency & Resolution Time Prediction (Regression)
* **Where the code lives:**
  - `src/grievanceroute/ml/regression.py`
  - `pages/04_ML_Workbench.py`

#### What it does:
Instead of predicting a category label, regression predicts **continuous numbers**:
1. **Urgency Score (1 to 5):** How critical is this problem?
2. **Estimated Resolution Duration (Hours):** How long will the repair take (e.g., 8 hours vs 36 hours)?

#### Which Model is Used?
- **Random Forest Regressor:**
  - An ensemble of 40 decision trees that vote together.
  - Features used: Zone, weather condition, time of day, whether it's monsoon, whether the area is low-lying, and the complaint text.
- **Compared Against a Naive Median Baseline:**
  - A dumb baseline that always predicts the median value (Urgency = 3, Time = 24h).
  - Our Random Forest significantly beats the baseline:
    - Urgency: Mean Absolute Error (MAE) drops to **0.7656** ($R^2 = 0.21$).
    - Resolution Time: MAE drops to **11.00 hours** ($R^2 = 0.17$).

---

### Step 5: Finding Geographic Hotspots (Unsupervised Clustering)
* **Where the code lives:**
  - `src/grievanceroute/ml/clustering.py`
  - `pages/03_Geographic_Hotspots.py`

#### What it does:
Helps municipal supervisors look at the entire map of Pune and answer two questions:
1. *"Where should we station our 5 regional repair depots?"* (Partitioning)
2. *"Where are dense corridors of complaints piling up right now, ignoring one-off isolated complaints?"* (Density)

#### Which Algorithms are Used?
1. **K-Means Clustering ($K=5$):**
   - Takes all complaint locations in projected kilometers $(x, y)_{\text{km}}$.
   - Partitions complaints into $K=5$ distinct clusters and computes the exact **centroid** (center point) for each.
   - Evaluated using **Silhouette Score = 0.4864** and **Inertia = 46,768.5**.
   - In the GUI, supervisors can move a slider to change $K$ from 3 to 8 dynamically.
2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
   - Uses radius $\epsilon = 0.8\text{ km}$ and minimum points $= 25$.
   - Any area with at least 25 complaints within 800 meters is declared a **dense corridor**.
   - Any complaint isolated by itself is labeled as **noise** (Cluster `-1`, 2.91% noise fraction), so the city doesn't send an entire emergency fleet for an isolated issue.

---

### Step 6: Pattern Mining Without Leakage (Apriori Algorithm)
* **Where the code lives:**
  - `src/grievanceroute/ml/association_rules.py`
  - `pages/07_Data_Quality.py` & `pages/06_Experiments.py`

#### What it does:
Like e-commerce systems find *"Customers who bought bread also bought butter"*, Apriori finds hidden correlations between civic conditions and complaints.
- Example discovered rule:
  $$\text{Monsoon = True} \land \text{Low-Lying Area = True} \implies \text{Complaint = Water Leak / Road Damage}$$

#### How it is implemented:
- Built **from scratch in pure Python** (candidate generation $C_k$, pruning, and support counting) and cross-validated with `mlxtend`.
- Calculates:
  - **Support:** How frequently the combination appears in the city.
  - **Confidence:** If condition A happens, how often does B follow?
  - **Lift:** How much more likely is B when A happens compared to random chance? ($\text{Lift} > 1$ means strong positive association).
- **Leakage Prevention:** Rules are mined strictly on environmental factors (zone, weather, monsoon, time) without using post-intake targets.

---

### Step 7: Road Pathfinding (A\* Search Algorithm)
* **Where the code lives:**
  - `src/grievanceroute/ai/search.py`
  - `src/grievanceroute/routing/road_graph.py`

#### What it does:
When a crew travels from depot to a complaint or between two complaints, they cannot fly in a straight line—they must travel on Pune's road network (16 major junctions like Swargate, Shivajinagar, Chandani Chowk, Katraj, Hadapsar). A\* finds the true shortest driving path.

#### How A\* Works ($f = g + h$):
- For any road junction node $n$:
  - $g(n)$: The actual road distance traveled so far from the start.
  - $h(n)$: The straight-line **Euclidean distance** from junction $n$ to the destination.
  - $f(n) = g(n) + h(n)$: The estimated total trip cost.
- **Why is Euclidean Distance an "Admissible" Heuristic?**
  - An admissible heuristic never overestimates the true distance ($h(n) \le h^*(n)$).
  - Because a straight line is the shortest possible distance between two points in geometry, actual road distance will always be greater than or equal to straight-line distance.
  - This mathematical guarantee proves that **A\* will always find the optimal shortest path**.

---

### Step 8: Multi-Crew Vehicle Routing (VRP & Local Search Optimization)
* **Where the code lives:**
  - `src/grievanceroute/routing/vrp_solver.py`
  - `src/grievanceroute/routing/crew_constraints.py`
  - `src/grievanceroute/ai/local_search.py`
  - `pages/05_Crew_Route_Planner.py`

#### What it does:
This is the core optimization engine. The city has **8 specialized repair crews** (2 Road Crews, 2 Water Crews, 2 Electrical Crews, 2 Sanitation Crews).
- Each crew has a depot, specialized skills, vehicle capacity, and a **strict 8-hour shift limit**.
- The system must assign complaints to the right crew and order the stops so total distance is minimized and **overtime is 0.0 hours**.

#### Global Objective Function $J(R)$:
$$J(R) = 1.0 \times \text{Total Distance (km)} + 10.0 \times \text{Unserved Urgency} + 5.0 \times \text{Total Overtime (Hours)}$$
The penalties for unserved high-urgency complaints ($\times 10$) and overtime ($\times 5$) ensure the algorithm prioritizes critical repairs without overworking the crews.

#### How the 3 Algorithms Work:
1. **Greedy Nearest-High-Priority (Baseline):**
   - Sorts complaints from highest urgency to lowest.
   - Assigns each complaint to the nearest available crew with matching skills and spare capacity.
   - *Result:* Quick initial schedule, but routes cross over each other (Total distance = 270.95 km).
2. **2-Opt Hill Climbing (Heuristic Search):**
   - Takes each crew's list of stops and tests swapping pairs of stops (2-opt move).
   - If swapping two stops untangles a loop and reduces travel distance, keep the swap.
   - Repeats until no more swaps improve the route.
   - *Result:* **194.85 km (28.09% distance improvement)**.
3. **Simulated Annealing (Stochastic Local Search):**
   - *The Problem with Hill Climbing:* Hill climbing is greedy—if all immediate neighbor swaps look slightly worse, it stops, even if a much better route is just two steps away (stuck in a local minimum).
   - *The Simulated Annealing Solution:* Inspired by how hot metal cools slowly to form a perfect crystal:
     - At high initial temperature $T$, it explores freely: if a swap is better ($\Delta E < 0$), it accepts it. If a swap is worse ($\Delta E > 0$), it still accepts it with Boltzmann probability $P = \exp(-\Delta E / T)$.
     - As temperature gradually cools ($T_{k+1} = 0.92 \times T_k$), it accepts fewer bad moves and settles into an optimal tour.
   - *Result:* **196.54 km (27.46% distance reduction) with 0.00 hours overtime**.

---

### Step 9: User Interface & Database Persistence
* **Where the code lives:**
  - `app.py`
  - `pages/01` to `07`
  - `src/grievanceroute/database/repository.py`
  - `src/grievanceroute/services/complaint_service.py`

#### What it does:
- Built with **Streamlit** (Python web framework) and **SQLite** (local database).
- All models, databases, and algorithms run **100% offline on standard CPU** (no cloud API keys, no paid tokens, no internet required).
- Provides 7 dedicated pages for operations supervisors, citizens, and examiners:
  - **01 Operations Dashboard:** Real-time KPI counters, urgent queue, category & zone charts.
  - **02 Complaint Management:** Interactive form to file complaints, run live triage, and advance ticket status (`Pending` $\rightarrow$ `Assigned` $\rightarrow$ `In Progress` $\rightarrow$ `Resolved`).
  - **03 Geographic Hotspots:** Interactive maps of Pune with K-Means centroids and DBSCAN density corridors.
  - **04 ML Workbench:** Compare models, ROC curves, confusion matrix, error analysis, and token explainability.
  - **05 Crew Route Planner:** Click a button to run Greedy vs 2-Opt vs Simulated Annealing and view crew tour maps.
  - **06 Experiments:** Export benchmark metrics and inspect hyperparameter runs.
  - **07 Data Quality & Insights:** View raw defects, mode imputation, MinMax/Standard scaling, sampling, and binning.

---

## 📊 4. Summary Table of Algorithms & Models

| Step in Pipeline | Problem Solved | AI / ML Method Used | File in Codebase | Performance / Result |
| :--- | :--- | :--- | :--- | :--- |
| **Intake Projection** | Map distortion | Equirectangular Metric Projection | `data/feature_engineering.py` | Converts GPS degrees to real km |
| **Safety Overrides** | Emergency hazards | Production Rule / Expert System | `services/prediction_service.py` | 100% deterministic catch of live wires |
| **NLP Triage** | Categorize complaint | Sublinear TF-IDF + Logistic Regression | `ml/classifier.py` | **96.37% Accuracy** (vs 26.87% baseline) |
| **Urgency Forecast** | Predict 1–5 urgency | Random Forest Regressor | `ml/regression.py` | **MAE 0.7656** (beats Median baseline) |
| **Duration Forecast** | Predict repair hours | Random Forest Regressor | `ml/regression.py` | **MAE 11.00 hrs** (beats Median baseline) |
| **Hotspot Centers** | Depot placement | K-Means ($K=5$) on projected km | `ml/clustering.py` | **Silhouette 0.4864**, Inertia 46,768.5 |
| **Dense Corridors** | High-density corridors | DBSCAN ($\epsilon=0.8\text{km}$, min 25) | `ml/clustering.py` | Filters 2.9% isolated spatial noise |
| **Pattern Mining** | Co-occurrence rules | Custom Apriori (Support/Conf/Lift) | `ml/association_rules.py` | Discovers monsoon/low-lying correlations |
| **Road Pathfinding** | Shortest junction path | A\* Search with Euclidean Lower Bound | `ai/search.py` | Admissible, optimal road paths |
| **Crew Scheduling** | Multi-crew VRP tour | 2-Opt Hill Climbing & Sim. Annealing | `routing/vrp_solver.py` | **~28% distance cut, 0.0h overtime** |
| **Persistence** | Data management | SQLite with CRUD & audit history | `database/repository.py` | ACID-compliant ticket lifecycle |

---

## 🎯 5. Quick 1-Minute Elevator Pitch (For Viva / Presentation)

> *"GrievanceRoute is an AI-driven civic operations platform for Pune Municipal Corporation. When a citizen files a complaint in plain English or Hinglish, the system first runs a deterministic safety check for acute electrical hazards like live wires. If safe, a TF-IDF Logistic Regression pipeline categorizes it with 96.4% accuracy, while Random Forest models predict its urgency and resolution duration. We project GPS coordinates into planar metric kilometers to detect spatial hotspots using K-Means and DBSCAN. Finally, our multi-crew routing engine uses A\* search, 2-Opt Hill Climbing, and Simulated Annealing to schedule 8 repair crews across the city—cutting total travel distance by 28% while guaranteeing zero overtime under strict 8-hour shift limits. The entire system is fully offline, reproducible, and tested with 21 automated pytest tests."*
