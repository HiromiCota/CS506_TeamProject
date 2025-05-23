# TP02 – Dance Move Classification Using Scikit-learn

This project is part of the CS506 Programming for Computing course. Our goal was to classify basic dance poses using movement-based features extracted from OpenPose-generated JSON files. The work demonstrates a supervised learning pipeline using Scikit-learn.

---

## 📂 Project Structure

```bash
.
├── sub2_dance_move_classifier.ipynb    # Full tutorial notebook with classification pipeline
├── sub2_dance_move_classifier.html     # Rendered export of the notebook
├── sub1_pose_utils.py                  # Pose loading & formatting module (Ixius's contribution)
├── pose_tools_byH.py                   # Feature vector generation & motion plotting (Hiromi's contribution)

🧩 Components Overview
📘 Notebook Highlights (sub2_dance_move_classifier.ipynb)
Extracts and scales pose movement vectors

Uses LabelEncoder and StandardScaler in a Scikit-learn pipeline

Classifies with both K-Nearest Neighbors (KNN) and Support Vector Machines (SVC)

Includes confusion matrix, classification report, PCA visualization

Visualizes pose transitions with vector arrows (Hiromi's tools)

🔧 Custom Modules
sub1_pose_utils.py
Created from work by Ixius to load pose keypoints, clean coordinates, and label samples from filenames sub1_pose_utils.

pose_tools_byH.py
Built from work by Hiromi to compute vectors between pose frames and visualize motion via quiver plot spose_tools_byH.

📊 Current Results
SVC Accuracy: ~66.7%, but heavily skewed toward the dominant class (5)

KNN Accuracy: ~11.1%, underperforms due to extreme label imbalance

Several warnings from Scikit-learn indicate undefined precision/recall on underrepresented classes

🛠️ Known Issues & Proposed Improvements
Problem	Potential Solution
Severe class imbalance	Apply class_weight="balanced" in models
Dominant class (5) overwhelms training	Downsample or cap class 5
Underrepresented labels get ignored	Use StratifiedShuffleSplit
Very small dataset	Add pose samples or simulate data with augmentation
Limited generalization	Explore external labeled pose datasets (e.g., Kaggle, CMU Mocap)

Acknowledgement:
OpenAI. (2025). ChatGPT’s assistance with Scikit-learn dance classification [Large language model]. https://openai.com/chatgpt