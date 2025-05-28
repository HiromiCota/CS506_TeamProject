# 💃 Submission #2 – Dance Move Classification Using Scikit-learn

This project is part of the CS506 Programming for Computing course. Our team’s goal was to classify basic dance poses using motion-based features extracted from OpenPose-generated JSON files. The notebook demonstrates a complete supervised learning pipeline implemented with Scikit-learn, enhanced with unsupervised learning for insight and validation.

---

## 📂 Project Structure

```bash
.
├── sub2_dance_move_classifier.ipynb     # Full tutorial notebook with classification pipeline
├── sub2_dance_move_classifier.html      # Rendered export of the notebook
├── pose_tools_byH.py                    # Movement vector generation and visualization (Hiromi's contribution)
├── sub1_pose_utils.py                   # Pose loading & preprocessing module (Ixius's contribution)
├── augment_poses.py                     # Standalone script to generate mirrored and jittered pose data
├── poses/                               # Folder containing all raw and augmented JSON pose data
├── README.md                            # This file
├── TP02-Presentation-links.md           # Links to prsentation slides and video

🔧 Custom Components
sub1_pose_utils.py (Submission #1)
Created by Ixius to:
- Load and clean JSON pose keypoints
- Build labeled DataFrames for Submission #1 compatibility
- Generate column labels and raw coordinate datasetssub1_pose_utils

pose_tools_byH.py (Submission #2)
Built by Hiromi to:
- Extract joint movement vectors across pose frames
- Generate pose transition visualizations using quiver plots
- Enable subplot-ready movement graphs using ax parameter supportpose_tools_byH

📊 Project Highlights
  🗂️ Pose Loading & Labeling: Used consistent filename prefixes (e.g., dance_01.json) and automated label extraction
  🔁 Augmentation: Used augment_poses.py to generate mirrored and jittered pose versionsaugment_poses
  🧹 Preprocessing: Cleaned missing data, encoded class labels with LabelEncoder, scaled features with StandardScaler
  🛠️ Feature Engineering: Switched from Submission #1-style coordinates to motion vectors using pose_list_to_vector_df()
  ✂️ Split: Used train_test_split() to evaluate model performance fairly
  🤖 Supervised Learning: Trained both KNN and SVC classifiers
  🧪 Model Evaluation: Analyzed performance with confusion matrices and classification reports (KNN: 25%, SVC: 21%)
  🛰️ Feature Space Visualization: Used PCA to inspect feature separability
  🌐 Unsupervised Learning: Applied both KMeans and GMM clustering to explore natural grouping of poses

🔍 Known Issues & Reflections
| Problem                              | Solution Implemented                          |
| ------------------------------------ | --------------------------------------------- |
| Severe class imbalance               | Applied `class_weight="balanced"` to SVC      |
| Underrepresented classes ignored     | Augmented poses with mirroring + jittering    |
| Some classes still not predicted     | Flagged for future tuning or alternate models |
| Limited generalization on small data | Proposed expansion with CMU Mocap / Kaggle    |

📌 Although KNN and SVC were implemented, future work should include hyperparameter tuning and deeper architectures (e.g., MLPs or LSTMs).

🧠 Key Takeaways
- Motion vectors are more informative than raw coordinates for classification.
- Balanced datasets require both augmentation and fair model training strategies.
- PCA and vector plots helped interpret both input data and model blind spots.
- Modularizing our Submission #1 and Submission #2 code helped us collaborate efficiently and iterate quickly.

📚 References
Matthes, E. (2023). Python Crash Course (3rd ed.). No Starch Press.
Microsoft. (2025). Fundamentals of Generative AI. Microsoft Learn. https://learn.microsoft.com:contentReference[oaicite:7]{index=7}
OpenAI. (2025). ChatGPT’s assistance with Scikit-learn dance classification [Large language model]. https://openai.com/chatgpt:contentReference[oaicite:8]{index=8}
Scikit-learn. (n.d.). User guide. https://scikit-learn.org/stable/user_guide.html:contentReference[oaicite:9]{index=9}
