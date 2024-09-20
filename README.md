# HDD-Failure-Prediction-Study
# Abstract
Hard disk drives are the most used component in storage drives; however, as with any other
device, they are also prone to failure. Hard disk failure prediction has become an active research
area. Recent trends have shifted towards using machine learning techniques using disk SMART
attributes to make predictions about disk failure. The extremely imbalanced Backblaze dataset has
been studied extensively with many methods, including feature selection and sampling techniques.
None of the previous studies analysed the effect of sampling ratios on the predictive model's
performance. Based on some of the best performing HDD analysis methods, our study intends to
verify their effectiveness on the newer Backblaze 2022 dataset and analyse the effect of sampling
ratios on the performance of predictive models. Bagging methods and new state-of-the-art
methods were studied and discussed in the study, along with the use of feature selection and
sampling techniques to address the challenges posed by highly imbalanced datasets. Bagging
classifiers were found to be more effective and consistent across different experiments. Sampling
techniques were found to be more effective in dealing with class imbalances than feature selection
methods.

# Terminology
BRF: Balanced Random Forest
BBC: Balanced Bagging Classifier
BBHG: Balanced Bag of Histogram Boosted Gradient Decision Trees
CMRR: Centre for Magnetic Recording Research
DT: Decision Tree
EE: Easy Ensemble
FAR: False Alarm Rate
FDR: False Detection Rate
FMMEA: failure modes, mechanisms, and effects analysis
G-mean / Gmean: Geometric Mean
GBDT: Gradient Boosted Decision Tree
GMM: Gaussian mixture model
HDD: Hard Disk Drives
IR: Imbalance Ratio
LSTM: Long Short-Term Memory
LSTM: Long Short-Term Memory
MD: Mahalanobis distance
PCA: Principal Component Analysis
RAT: Reverse Arrangement Test
RF: Random Forest
RFE: Recursive Feature Elimination
RNN: Recurrent Neural Network
ROS: Random Over Sampling
RUS: Random Under Sampling
SMART: Self-Monitoring, Analysis and Reporting Technology
SMOTE: Synthetic Minority Over Sampling Technique
SVM: Support Vector Machine
TIA: Time in Advance
TLDFP: Transfer Learning for Minority Disk Failure Prediction
WLR: Weighted Logistic Regression


# INTRODUCTION
## Background
Hard disk drives (HDD) have come a long way since their development around 70 years ago.
Initially, manufacturers never anticipated the immense success that these devices would achieve in
modern times, as now they are used to store critical data almost everywhere you look. One key
reason for HDDs' surge in popularity and wide-spread usage is due to their impressive durability
compared to other hardware components like RAM or flash memory which can easily succumb
electrostatic discharge whereas HDDs are resistant to electromagnetic interference ensuring no
loss of stored information. While hard disk drives (HDDs) are generally considered reliable, they
have earned a reputation as the most frequently replaced hardware component in personal
computers (PCs) (Li et al. 2014). Despite having a history of failure, the incidents of HDD
malfunctions have risen significantly (Schroeder and Gibson 2007b). As per reports, a staggering
78% of Microsoft data centres' hardware replacements were attributed to hard drives (Vishwanath
and Nagappan 2010). Therefore, it is a vital and active research area to improve the reliability and
availability of data storage systems.
A hard disk drive (HDD) is a series of disks that use one or more rigid, rapidly rotating disks to
store data. The drive is responsible for data storage. There are four main units of an HDD: One of
the components of an HDD is the Head disk interface. The other three include head stack
assembly, spindle motors/bearings, and the electronics module (Wang et al. 2011; Shen et al.
2018). A hard disk's mechanical and electrical parts will each have different early-warning signals
for when a failure is about to happen. For example, the head–disk interface is an excellent place to
look for early warning, as is the quality of the motor or bearings. The types of problems that will
affect these parts are also different, so each aspect of the disk will give you some information to
pick out from. A head–magnetization problem will be very different to a read error. By using these
separate pieces of information and separating them out, you get a much clearer picture of the
health of the drive (Shen et al. 2018).
There are three major categories of algorithms and techniques that have been applied by past
research into hard drive failure prediction: The first one is to try to predict the amount of remaining
time until failures of the chosen drives occur as a forecasting problem (Lima et al. 2018; Züfle et al.
2020). An approach to HDD failure prediction is as a series of events outside of the control of the
drive and then to predict the probability of such events based on current and historical information
about the drive. Specifically, historical data about the average lifetime of a given disk is used to
build a regression model, which can be used to predict the expected lifetime of a given disk.

However, both the difficulties involved in obtaining sufficient failure of the disk examples and the
fact that their parameters depend on the operating conditions and environment make it challenging
to learn a distinctive model (Ircio et al. 2022b). Still, other algorithmic approaches include machine
learning and clustering.
The second strategy involves modelling the drive's health condition as a multi-class classification
issue (Liu et al. 2020; Züfle et al. 2020). Consequently, the multiple classes are defined as varying
health statuses to differ in consequence of the failure over time. The highest degree indicates
explicitly that the disc is functioning properly, and for the other grades, the lower the grade, the
closer failure is, with the lowest grade signifying impending failure. It should be noted that the
previously recommended technique is equivalent to discretizing a previously described regression
model and therefore possesses the same difficulties. Additionally, because failures are rare, the
percentage of successful and unsuccessful discs is quite imbalanced, and by producing additional
classes, this issue is even worse (Ircio et al. 2022a).
The third and most popular method models the hard drive state binary classification problem as the
failure prediction problem (correct or failed) (Aussel et al. 2017; Queiroz et al. 2017; Ahmed and
Green 2022). A disk is classified as failed if it exhibits signs of deterioration, indicating that it will fail
within a short period. We will be using this approach in our study. While this is the most preferred
approach, it does have its flaws. This is because the failure lead time is known in advance. It is
assumed that all drives will show signs of malfunction with exactly the same anticipation. The
problem is that there is no one lead-time that works for all hard drives. This is because the
environment in which the drive operates, and its life expectancy is different from each other.
Therefore, an automated method is developed to detect every disk's first signs and symptoms of
malfunction. It does not require that each disk be pre-set with a lead time.
In this research, we reviewed some of the best-performing HDD failure analysis methods that have
been used in earlier research. Our study focuses on verifying whether existing HDD methods are
still a good fit for predicting HDD failures, as the dataset is getting imbalanced every year. We
compared the methods using the latest 2022 Backblaze dataset and various class imbalance
techniques to verify and in the hope of further improving performance. We also measured the
effect of different sampling ratios on the machine learning models.
We thoroughly review existing HDD failure prediction methods, highlighting their strengths and
limitations. Next, we verified those methods on the newer dataset that leverages advanced
machine learning techniques, such as ensemble methods and hybrid methods, to improve the
accuracy and reliability of HDD failure predictions. To further improve those existing methods, we
used techniques such as feature selection and sampling methods.

Finally, we compared the effect of different sampling ratios on the performance of ML models.
Based on our analysis of the Backblaze HDD dataset, sampling techniques are more important
than feature selection when it comes to resolving the issue of class imbalance in the dataset.

## SMART
The SMART (Self-Monitoring, Analysis and Reporting Technology) system is a monitoring system
included in computer hard disk drives (HDD) and solid-state drives (SSD) that detects and
analyses various indicators of reliability in the hope of anticipating failures. The SMART system
was implemented on all ATA and SATA drives and is also supported by SCSI and SAS interfaces.
Using SMART, built-in functions for HDDs that collect data that correspond to records or physical
units by sensors or counters (Hughes et al. 2002; Pinheiro et al. 2007). Up to thirty internal drive
attributes such as relocated sector count (RSC), spin-up time (SUT), seek error rate (SER),
temperature Celsius (TC), and power-on hours (POH) are included in SMART data (Shen et al.
2018). These attributes are related to the hard disk drive's health and can be used to determine its
internal state. The value of relocated sector count (RSC) indicates the number of bad sectors. A
change in the Temperature Celsius (TC) and spin-up time (SUT) strongly corresponds with the
health condition of the spindle motor.
The five fields that make up each attribute are raw data, value, threshold, worst value, and status.
The values measured by a sensor, or a counter are raw data. The value is the current raw data's
normalized value. Failure is detected using the threshold. HDD manufacturers specify the
threshold value and the algorithm for calculating the value. A warning is issued when the
normalized value goes above the threshold. Additionally, SMART will sound a failure alarm when
any attribute's status changes to a warning (Shen et al. 2018).
Drive failures are significantly associated with SMART characteristics. Three attributes—grown
defect count, read soft errors, and seek errors—were combined in a prediction technique provided
by (Hamerly and Elkan 2001). This technique produced predictions with higher accuracy when all
three attributes were combined. According to this, not all SMART qualities are equally helpful for
predicting HDD failure. (Pinheiro et al. 2007) Found that relocated sector count, scan error and
offline relocated sector count errors correlate highly with HDD failures. According to (Ma et al.
2015), RSC is the most crucial factor for identifying impending failures, with latent sector faults
being the primary cause of HDD failure. (Wang et al. 2013) identified different indicators of failure
in inner units of the HDD and determined the priority of SMART attributes to be used in a prediction
system based on the severity and occurrence of relevant failures. (Huang 2017) showed that
failure-correlated attributes of SMART for each type of HDD failure are pretty different. The failure
prediction of HDDs is therefore based on strong-correlation attributes rather than a single
combination.
Data centres commonly use hard disk drives as data storage devices, but as the number of hard
drives used in the storage systems increases, it becomes more challenging to maintain their
reliability (Schroeder and Gibson 2007a; Li et al. 2014; Shen et al. 2018). When hard disk drives
fail, it can lead to information losses, which can be a considerable problem for the users. Even
though multiple copies of data can be stored in the system as backup, it can also increase costs at
the same time (Huang 2017). According to Backblaze, the Annual Failure Rate (AFR) for all the
drives in 2021 was 1.01% which is around 1820 drives out of 202,759 (Klein 2022). Hard disk drive
failure is a prevalent problem. This may happen in any area of the disk drive, and therefore it is not
possible to predict with certainty where a failure will occur. In the case of a single drive, a
replacement is usually sufficient. A disk mirroring, disk duplexing, or disk striping system can be
used as a contingency measure in case of failure of one of two or more drives respectively. In such
cases, the mechanisms provided by the operating system can be used to replace one of the drives
without requiring any manual configuration changes. However, this solution requires buying a
second hard disk drive. This expense might be prohibitive for small businesses, and a more
straightforward solution might be desirable. Consequently, it seems practical to develop a model
that can predict hard drive failure and then the operators can use the prediction result to improve
system reliability and reduce cost.
The most common cause of hard drive failure is the accumulation of errors on the magnetic media.
Hard drives are designed to compensate for these errors by writing data to a different location on
the disk. However, when the number of errors exceeds a certain threshold, the drive will no longer
be able to compensate, and data will be lost (Schroeder and Gibson 2007a). The number of errors
that a hard drive can compensate is limited by the number of spare sectors available on the disk.
Spare sectors are used to replace sectors that have become unusable due to excessive errors.
When a hard drive is new, it has many spare sectors available. Hard disk drives have a finite
number of read and write operations, after which failure is inevitable. A hard disk drive is built to
withstand a single component's failure but not multiple components within the same component
category. If the head fails, spare heads are on the drive to take their place. However, if a head and
a platter both fail, the drive is rendered inoperable (Aussel et al. 2017). Many predictive models
have been proposed to mitigate hard drive failures (discussed in chapter 2.4) but they all have their
own limitations. For example, some models need to be more accurate to accurately predict the
time of failure, while others cannot be used in real-time systems because they require too much
computation power. Based on what has been presented so far, the diagnosis of hard disk drive
failure should be well on its way to being solved. However, it remains a problem. Why is this the
case? The best explanation for this is that hard disk drive failure has been addressed as a problem
dictated by a single failure mode (e.g., head crash). However, more is needed to explain why the
high performance of the failure prediction models that appear in the literature has yet to mitigate
the problem. Therefore, the specifics of hard drives need to be better taken into account: First,
because hard drives are assembled from components with different performances, it is necessary
to quantify their individual performances to have a final performance expressed in a single value.
Second, as hard drive failure only results from a single cause, failure prediction models must be
based on something other than event counts per sector. Instead, they should be based on events
weighted by their relevance. Furthermore, A class imbalance is a common issue in the research
domain, where the number of healthy hard disks greatly exceeds the number of failing ones.
Therefore, this study will investigate and evaluate methods for handling this imbalance.
## Aims and Research Questions
Our project aims to investigate the possibility of applying machine learning techniques to improve
prediction accuracy over baseline HDD methods in hard disk drives. Our main objectives are:
1. Review existing HDD methods: pre-processing, feature selection methods and ML
methods.
2. Evaluate one HDD manufacturer using the latest Backblaze dataset, utilizing machine
learning to predict failure with imbalanced data.
3. Compare and evaluate different models using the latest Backblaze dataset, which includes
hard drives from various manufacturers.
4. Compare the ML models while using different sampling ratios to see how they affect the
given model.
5. Discover which potential pre-processing methods may further improve the performance.
In order to narrow the project's scope, three research questions have been formulated:
1. Which data pre-processing technique can enhance performance on analysing Backblaze
2022 datasets, and how do they compare to existing methods?
2. What are the differences in performance when analysing Backblaze 2022 datasets using
the Recursive Feature Elimination (RFE), Random Under-Sampling (RUS), and Balanced
Random Forest (BRF) methods on imbalanced datasets?
3. How does the sampling ratio affect the performance of proposed method when analysing
Backblaze 2022 datasets?

## The structure of the thesis
Chapter 1: Introduction – This chapter considers prior HDD failure prediction options, defines the
issue, and provides background information. We also examine SMART attributes within HDDs and
describe our research’s objectives and goals.
Chapter 2: Literature Review – This chapter examines numerous techniques for handling
imbalanced datasets including sampling methods, feature selection methodologies, plus
performance measures alongside relevant literature overviews covering these areas extensively.
Chapter 3: Methodology – This chapter defines our methodology, evaluates potential risks
involved, and identifies necessary resources for successfully completing this work utilizing
adaptations from CRISP-DM frameworks according to what fits best within current requirements.
Chapter 4: Experiments – This chapter discusses about conducting experiments utilizing
Backblaze data to provide an overview of results with comparisons between different tested
methodologies.
Chapter 5: Discussion – This chapter discusses the experiments of this research, and the
objectives will be critically analysed. This chapter also discusses limitations plus possibilities for
future research beyond its current focus.
Chapter 6: Conclusion – This chapter concludes this study by summarizing key findings and then
outlining their implications for HDD failure prediction.
