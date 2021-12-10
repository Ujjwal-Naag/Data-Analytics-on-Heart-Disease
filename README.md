# Data-Analytics-on-Heart-Disease
## Heart disease prediction using Machine Learning
According to the World Health Organization, every year 12 million deaths occur worldwide due
to Heart Disease. The load of cardiovascular disease is rapidly increasing all over the world from
the past few years. Many researches have been conducted in attempt to pinpoint the most
influential factors of heart disease as well as accurately predict the overall risk. Heart Disease is
even highlighted as a silent killer which leads to the death of the person without obvious symptoms.
The early diagnosis of heart disease plays a vital role in making decisions on lifestyle changes in
high-risk patients and in turn reduce the complications. This project aims to predict future Heart
Disease by analyzing data of patients which classifies whether they have heart disease or not using
machine-learning algorithms.
Every day, the average human heart beats around 100,000 times, pumping 2,000 gallons of blood through the body. Inside your body there are 60,000 miles of blood vessels.

The signs of a woman having a heart attack are much less noticeable than the signs of a male. In women, heart attacks may feel uncomfortable squeezing, pressure, fullness, or pain in the center of the chest. It may also cause pain in one or both arms, the back, neck, jaw or stomach, shortness of breath, nausea and other symptoms. Men experience typical symptoms of heart attack, such as chest pain , discomfort, and stress. They may also experience pain in other areas, such as arms, neck , back, and jaw, and shortness of breath, sweating, and discomfort that mimics heartburn.

It’s a lot of work for an organ which is just like a large fist and weighs between 8 and 12 ounces.

## Code
### Github Link : https://github.com/Ujjwal-Naag/Data-Analytics-on-Heart-Disease

## Datatset
### Dataset Link :  https://archive.ics.uci.edu/ml/datasets/Heart+Disease

## Introduction
The major challenge in heart disease is its detection. There are instruments available which can
predict heart disease but either they are expensive or are not efficient to calculate chance of heart
disease in human. Early detection of cardiac diseases can decrease the mortality rate and overall
complications. However, it is not possible to monitor patients every day in all cases accurately and
consultation of a patient for 24 hours by a doctor is not available since it requires more sapience,
time and expertise. Since we have a good amount of data in today’s world, we can use various
machine learning algorithms to analyze the data for hidden patterns. The hidden patterns can be
used for health diagnosis in medicinal data. Machine learning techniques have been around us and has been compared and used for analysis
for many kinds of data science applications. The major motivation behind this research-based
project was to explore the feature selection methods, data preparation and processing behind the
training models in the machine learning. With first hand models and libraries, the challenge we
face today is data where beside their abundance, and our cooked models, the accuracy we see
during training, testing and actual validation has a higher variance. Hence this project is carried
out with the motivation to explore behind the models, and further implement Logistic Regression 
2
model to train the obtained data. Furthermore, as the whole machine learning is motivated to
develop an appropriate computer-based system and decision support that can aid to early detection
of heart disease, in this project we have developed a model which classifies if patient will have
heart disease in ten years or not based on various features (i.e. potential risk factors that can cause
heart disease) using logistic regression. Hence, the early prognosis of cardiovascular diseases can
aid in making decisions on lifestyle changes in high risk patients and in turn reduce the
complications, which can be a great milestone in the field of medicine.

#### Objective
The main objective of developing this project are:
1. To develop machine learning model to predict future possibility of heart disease by
implementing Logistic Regression.
2. To determine significant risk factors based on medical dataset which may lead to heart
disease.
3. To analyze feature selection methods and understand their working principle.

## Introduction to the Dataset

So here it goes, the description of the dataset.

cp: chest pain type -- Value 0: asymptomatic -- Value 1: atypical angina -- Value 2: non-anginal pain -- Value 3: typical angina

restecg: resting electrocardiographic results -- Value 0: showing probable or definite left ventricular hypertrophy by Estes' criteria -- Value 1: normal -- Value 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)

slope: the slope of the peak exercise ST segment 0: downsloping; 1: flat; 2: upsloping

thal Results of the blood flow observed via the radioactive dye.

Value 0: NULL (dropped from the dataset previously) Value 1: fixed defect (no blood flow in some part of the heart) Value 2: normal blood flow Value 3: reversible defect (a blood flow is observed but it is not normal) This feature and the next one are obtained through a very invasive process for the patients. But, by themselves, they give a very good indication of the presence of a heart disease or not.

target (maybe THE most important feature): 0 = disease, 1 = no disease

A few more things to consider: data #93, 139, 164, 165 and 252 have ca=4 which is incorrect. In the original Cleveland dataset they are NaNs (so they should be removed) data #49 and 282 have thal = 0, also incorrect. They are also NaNs in the original dataset.

It's a clean, easy to understand set of data. 
However, the meaning of some of the column headers are not obvious. 
Here's what they mean, 
• age: The person's age in years 
• sex: The person's sex (1 = male, 0 = female) 
• cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic) 
• trestbps: The person's resting blood pressure (mm Hg on admission to the hospital) 
• chol: The person's cholesterol measurement in mg/dl 
• fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false) 
• restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria) 
• thalach: The person's maximum heart rate achieved 
• exang: Exercise induced angina (1 = yes; 0 = no) 
• oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here) 
• slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping) 
• ca: The number of major vessels (0-3) • thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect) 
• target: Heart disease (0 = no, 1 = yes)

##### Description of the Dataset

This is multivariate type of dataset which means providing or involving a variety of separate mathematical or statistical variables, multivariate numerical data analysis. It is composed of 14 attributes which are age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise induced angina, oldpeak — ST depression induced by exercise relative to rest, the slope of the peak exercise ST segment, number of major vessels and Thalassemia. This database includes 76 attributes, but all published studies relate to the use of a subset of 14 of them. The Cleveland database is the only one used by ML researchers to date. One of the major tasks on this dataset is to predict based on the given attributes of a patient that whether that particular person has a heart disease or not and other is the experimental task to diagnose and find out various insights from this dataset which could help in understanding the problem more.

##### Dataset was Created By: 
1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.


### 1. Imports and Reading Dataset
