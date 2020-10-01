# Motivation
This project is to complete the capstone project of Udacity Data Scientist Nanodegree, analyzing customer segmentation of a mail-out order company in German.

In this project, I have performed unsupervised and supervised learnig to answer following two questions:

1. Who are the loyal customers of the company, and with a change in marketing strategy to expand customer demographics, who are the potential customers to target?

2. When the company sends out a mailout offer, can we predict the responding rate?

The details of my analysis can be found in blog post ([Medium](https://medium.com/@wenzhili523/airbnb-market-in-seattle-a-closer-look-from-the-perspectives-of-host-and-visitor-89b179cde17))

# Required Libaries
This project was written by Python, using Jupyter Notebook, including common packages: 
* pandas 
* numpy 
* matplotlib 
* seaborn
* sklearn
* clean (included in the descrition files)

# The Description of Files
1. Arvato Project Workbook.ipynb — includes the written code for data analysis
2. clean.py — data wrangling function
3. feature_summary.csv — dataset includes the description of each features 

(Note: the raw datasets were limited to Udacity and Arvato, so not provided in this repository)

# Summary

By unsupervised learning, I successfully grouped the original datas into eight clusers: 
* Cluster 7 is the core customer base of the company, the individuals in this cluser are more likely Middle-class with small families and lively online shopping habits.
* Cluster 2 is the potential new customer candidates, the individuals in this cluster are upper class, who are fascinating by German luxury car and originated from West-Germany. 

By building a supervised machine learning model, I predict the mail-out offer responding rate with relatively satisfied score:
* AdaBoost algorithms can achieve ROC AUC score of 0.781 in kaggle competition
* GradientBoost algorithm can achieve ROC AUC score of 0.777 in kaggle competition

# Credit and Reference
Data and resource provided by [Udacity](https://www.udacity.com/) and [Bertelsmann&Arvato](www.bertelsmann.com/divisions/arvato/#st-1)
