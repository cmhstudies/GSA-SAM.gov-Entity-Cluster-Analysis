# GSA SAM.gov Entity Cluster Analysis
 Investigating the applicability of clustering algorithms to identify similar government contractors.

## Motivation

While supporting business development and proposal preparation activities for small business Federal Government contractors, a frequent question arises regarding the identification of similar and dissimilar businesses. Federal contractors use this information to identify competitors. This leads to additional research and strategic decisions about selection of teammates and solution design.

In pursuit of a Masters of Science in Data Analytics (MSDA) from Western Governors University (WGU), the final capstone project requires application of skills and knowledge gained throughout the program to an analysis project of the student's design. This repository contains the Python code and Jupyter Notebooks used for my capstone project. The project proposes using publicly-available entity registration data for investigating the applicability of clustering techniques.

**Note:** Due to feedback from my capstone instructor and the nature of this problem falling more in the category of Data Science versus Data Analytics, I chose an alternative topic for my capstone project. See [Predicting Incident Managment SLA Failures](https://github.com/cmhstudies/Incident-Management-Process-BPIC2014). The application of cluster analysis techniques still interests me and development may continue. 

## Data Source

The System for Award Management (SAM), located at [www.sam.gov](http://www.sam.gov/), holds information about entities (individuals, companies, and organizations) registered to conduct business with the U.S. Federal Government. The U.S. General Services Administration (GSA) maintains SAM and releases a monthly extract of publicly available entity registration data available at [https://sam.gov/SAM/pages/public/extracts/samPublicAccessData.jsf](https://sam.gov/SAM/pages/public/extracts/samPublicAccessData.jsf). The [SAM Entity Management Public Extract Layout](https://sam.gov/SAM/transcript/SAM_Entity_Management_Public_Extract_Layout_v1.1.pdf) and the [SAM Functional Data Dictionary](https://gsa.github.io/sam_api/sam/SAM_Functional_Data_Dictionary_v7_Public.pdf) documents describe the file format. 

## Challenges

### Dealing with Categorical Data

The data set contains strictly categorical data while clustering algorithms require numeric data. 

## Questions

While developing the project proposal, the following questions arose: 
- Does this data set contain enough distinguishing information for clustering algorithms to identify similar and dissimilar organizations? 
- What measures will assist in determining the effectiveness of the clustering algorithms? 
- Will the algorigthms provide more insightful information than a subject matter expert's (SME) opinion? 

   - How would one determine if the algorithms produced more insightful information than and SME's opinion?

## Data Analysis Steps

[1. Read and Filter Source Data](https://github.com/cmhstudies/GSA-SAM.gov-Entity-Cluster-Analysis/blob/master/1.%20Read%20and%20Filter%20Source%20Data.ipynb)

[2. Initial Exploratory Data Analysis](https://github.com/cmhstudies/GSA-SAM.gov-Entity-Cluster-Analysis/blob/master/2.%20Initial%20Exploratory%20Data%20Analysis.ipynb)

3. To Be Determined

4. To Be Determined
