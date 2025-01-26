## Goal
To segment mall customers into distinct groups based on their spending behavior and leverage these insights for targeted marketing strategies aimed at improving sales and customer engagement.

## Process

I performed exploratory data analysis (EDA) to understand spending patterns, applied K-Means clustering with the elbow method to determine optimal clusters, and conducted bivariate and multivariate analyses. To ensure unbiased results, I used feature scaling and standardization.

## Data Source:
https://www.kaggle.com/code/suneelpatel/mall-customers-segmentation-clustering-model/data

# Findings

- Result 1: The customer base is dominated by younger individuals who form the core demographic for analysis. Most belong to a middle-income group, while spending patterns reveal two key segments: value-conscious customers and high-value spenders, providing actionable insights for targeted segmentation strategies.
    - The majority of customers earn between 50k and 90k, with a decline in the proportion for higher incomes beyond 100k.
    - The Spending Score distribution indicates two clusters: one in the lower spending range around 10-50 and another around 60-100, suggesting distinct spending behaviors within the customer base.
    - The Age distribution shows a younger-dominated user base, with most customers aged between 20 and 40. Older age groups (above 50) form a smaller proportion of the mall’s customer base.

https://amber-giraffe-fc3.notion.site/image/attachment%3A6b6848a9-ae9e-4195-bddf-9f3aec5c565f%3Adistribution_Plot_Mall_Customers_Income.png?table=block&id=187e1ad3-8135-80f1-b6c0-e71e14ae7cfc&spaceId=a1d8b709-8be6-4876-9027-347918b52a9c&width=410&userId=&cache=v2

https://amber-giraffe-fc3.notion.site/image/attachment%3Aa78a437a-e6b5-4aca-88d8-47ba7805763e%3Adistribution_Plot_Mall_Customers_Spend_Score.png?table=block&id=187e1ad3-8135-8000-8819-fa74a9e2c7d7&spaceId=a1d8b709-8be6-4876-9027-347918b52a9c&width=2000&userId=&cache=v2

https://amber-giraffe-fc3.notion.site/image/attachment%3A6c1b18b4-341b-4e3d-bbef-189a272e8cd8%3Adistribution_Plot_Mall_Customers.png?table=block&id=187e1ad3-8135-80a5-afd0-ee698179b773&spaceId=a1d8b709-8be6-4876-9027-347918b52a9c&width=2000&userId=&cache=v2

- Result 2: The data highlights distinct customer segments based on gender, age, income, and spending patterns, providing actionable insights for targeted marketing. Younger customers tend to exhibit higher spending potential, particularly among females, while older customers display more conservative spending behaviors. Clear groupings of customers emerge across income and spending levels, allowing for focused segmentation to optimize promotional efforts and campaign strategies.
    - Females make up a slightly larger proportion (56%) of the user base than males. Most males are younger (20-50), while females' ages extend into the 70s.
    - Younger users typically earn low-to-middle incomes. As age increases, incomes shift to upper-middle and higher levels.
    - Two key age-spend groups emerge: young high spenders (dominated by females) and older users (40s-70s) clustered around low-to-average spend scores.
    - Income-spend segmentation reveals five groups: Low Income-Low Spend, Low Income-High Spend, Middle Income-Average Spend, High Income-Low Spend, and High Income-High Spend.

https://amber-giraffe-fc3.notion.site/image/attachment%3A0d204e02-f8e5-49f2-ae43-ec070ce0d046%3AGender_Distribution_wrt_income_spend_and_age.png?table=block&id=187e1ad3-8135-80d2-b812-d8090ea96861&spaceId=a1d8b709-8be6-4876-9027-347918b52a9c&width=2000&userId=&cache=v2

- Result 3: I have identified five distinct customer groups: two in low-income, two in high-income, and one in middle-income with average spending. Notably, both high-spending groups are predominantly younger individuals, while spending behaviors and income levels define below listed actionable strategies. which ensures targeted interventions to maximize sales and foster long-term customer loyalty.
    - **High-Spending, Low-Income Segment (Predominantly Women):** This group presents a strong opportunity, but strategies should focus on improving the quality of products they purchase or incentivizing them through targeted discounts and loyalty programs to boost sales sustainably.
    - **High-Spending, High-Income Segment (Younger):** As premium customers, this group should receive loyalty benefits or subscription-based offers tailored to their preferences. Identifying luxury or high-end products that cater to their needs can increase engagement and maximize sales.
    - **Low-Spending, High-Income Segment (Predominantly Men):** This group has potential for conversion into high-value customers. Strategies should include identifying items they currently purchase and targeting them with advertisements for age-appropriate or aspirational products to encourage increased spending.

https://amber-giraffe-fc3.notion.site/image/attachment%3Af2fc07f0-9871-4761-9791-eecc8a378dc0%3AMall_Customers_Segments.png?table=block&id=187e1ad3-8135-8053-9c13-ddf58d301059&spaceId=a1d8b709-8be6-4876-9027-347918b52a9c&width=750&userId=&cache=v2

https://amber-giraffe-fc3.notion.site/image/attachment%3A79d4fb5a-e3cb-4541-b259-a130c610e4a6%3AMall_Customers_Segments_Mean_Stats.png?table=block&id=187e1ad3-8135-8085-9ae6-dc6fa22fd776&spaceId=a1d8b709-8be6-4876-9027-347918b52a9c&width=580&userId=&cache=v2

https://amber-giraffe-fc3.notion.site/image/attachment%3A9269db36-995a-415a-816c-5a6cfe1784c4%3AMall_Customers_Segments_Gender_Composition.png?table=block&id=187e1ad3-8135-8004-9821-db8cbe211761&spaceId=a1d8b709-8be6-4876-9027-347918b52a9c&width=2000&userId=&cache=v2


Complete Overview of the project: https://medium.com/@altamashakber/k-means-clustering-for-customer-segmentation-ef7981619475
