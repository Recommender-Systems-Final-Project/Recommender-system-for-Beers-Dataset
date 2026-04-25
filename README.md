# Recommender-system-for-Beers-Dataset

Edit Required Before Submission

This README is specifically designed for Individual Variant contribution (Sai Chetan) and explains how to integrate your work with the Shared Foundation.

Beer Recommender: Advanced Hybrid & Cold-Start Variants
Project Overview
This repository contains the individual advanced variants for the Beer Recommendation System project. These models build upon a shared foundation of 3.3 million reviews from BeerAdvocate and RateBeer to solve complex recommendation challenges: Data Sparsity (Cold-Items) and The Filter Bubble (Cold-Users).

Core Methodologies
HybridNeuMF (Neural Matrix Factorization): A deep learning architecture combining Generalized Matrix Factorization (GMF) and a Multi-Layer Perceptron (MLP) to fuse user-item interactions with beer metadata.

Bayesian Hybrid Logic: A multi-stage inference policy that dynamically weights global popularity against content affinity based on user history size.

MMR Diversity Re-ranking: An implementation of Maximal Marginal Relevance to ensure high style diversity in top-N recommendations.

File Descriptions
1. Beer_Recommender_Hybrid_SAS.ipynb
Purpose: The primary Training & Model Design engine.

Contents:

Architecture definitions for HybridNeuMF and MetadataSASRec.

Feature engineering pipeline for ABV normalization and Sensory Aspect (Aroma, Taste, etc.) extraction.

Training loops and model checkpointing (.pth files).

Use Case: Run this if you want to modify the neural network weights or retrain on updated data.

2. Beer_Recommender_LoadAndEval_v3.ipynb
Purpose: The Inference & Optimization standalone notebook.

Contents:

Loads pre-trained weights without requiring the training hardware.

Implements the v3 Cold-Start fix (Bayesian weighting).

Contains the recommend_hybrid function with Rarity Penalties and MMR Diversity.

Use Case: This is the primary notebook for the Oral Presentation demos and final evaluation metrics.

Advanced Baselines Used
Beyond the standard User/Item/Rating interactions, this project implements several sophisticated baselines to improve utility:

Bayesian Style-Priors: Instead of a raw mean rating, we use a Bayesian damped mean to weight style popularity. This prevents "niche" styles with 1 review from unfairly dominating the rankings.

Log-Count Popularity: An item-frequency baseline used to normalize "Beer Fame." This helps the model distinguish between a beer that is good because it is popular and a beer that is good because it matches the user's taste.

ABV Affinity: A content-based baseline that calculates the Euclidean distance between a user's average preferred ABV and the candidate beer's ABV.

Style-Clumping Penalty (MMR): A diversity baseline that checks the entropy of the Top-10 list. If more than 3 beers of the same style appear, the system applies a penalty to the next similar item to force discovery.

How to Run
Prerequisites
Environment: Google Colab (Pro recommended for Hybrid_SAS, Standard is fine for LoadAndEval).

Data: Ensure the processed df_combined.parquet and the saved .pth model weights are uploaded to your Google Drive.

Execution Steps
Mount Drive: Run the first cell to link your Google Drive.

Initialize Metadata: Run the Data Loading section. This reconstructs the user2idx and item_style dictionaries required for the models to understand the beer IDs.

Model Loading: Execute the Architecture Definition cells followed by model.load_state_dict().

Inference:

To test a specific user: Set target_user = 'Lagerboy' and run the recommendation cell.

To run the full evaluation: Run the Ranking Evaluation section to see Hit Rate (HR@10) and NDCG metrics.

Key Performance Indicators (KPIs)
Cold-Item Accuracy: HybridNeuMF provides a 9.8% RMSE improvement over the global mean for items with few ratings.

Including Models in Github for reference, Just load models in Drive and Run LoadAndEval Notebook


Cold-User Utility: The Bayesian Hybrid fallback narrowed the cold-start performance gap from -44.4% to -4.8% against a popularity baseline.

Diversity Score: The MMR logic consistently maintains 90% style diversity (9 distinct styles) in Top-10 lists, even for style-specialist users.
