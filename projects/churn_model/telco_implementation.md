Yes. Churn is the most important thing. It is the heartbeat of everything in this role.

---

**Description of the Telco Churn Model**

A churn prediction model is a supervised machine learning system that identifies which prepaid subscribers are at risk of becoming inactive before they actually go silent. In prepaid telecom there is no cancellation event — customers simply stop topping up and disappear. The model's job is to detect the behavioural signals of that drift early enough for the business to intervene.

The output is a probability score between 0 and 1 for every active subscriber. A score of 0.85 means that customer has an 85% likelihood of churning within the defined window — typically 30, 60, or 90 days. That score then feeds a targeting engine that decides who gets a retention offer, what offer they receive, and through which channel.

The business value is straightforward. Retaining an existing customer costs a fraction of acquiring a new one. If your model correctly identifies even 60% of customers who would have churned, and a retention offer saves half of those, the incremental revenue protected can be in the tens of millions across a large subscriber base.

---

**Implementation Plan**

**Phase 1 — Problem Definition**

Before touching data you define the business problem precisely. You decide on the churn definition — in prepaid this is typically 90 days of no revenue-generating activity. You define the prediction window — you want to predict churn 30 days in advance so the business has time to act. You define what a successful model looks like in business terms — not just AUC-ROC but how many churners caught, at what intervention cost, and what retention rate is needed to break even.

This phase grounds everything that follows. A model without a clear business definition is just an exercise.

---

**Phase 2 — Data Collection and Understanding**

You identify what data exists and what it means. In telecom this is primarily CDR data — Call Detail Records — which captures every network event per subscriber. From this you extract top-up history, data consumption, voice usage, SMS activity, bundle purchases, and channel behaviour over time.

You spend time understanding the data before engineering anything. How far back does it go? Are there gaps? Are there known data quality issues? What does a typical active customer look like versus a customer who churned historically? This exploration shapes every decision in the phases that follow.

---

**Phase 3 — Feature Engineering**

This is where most of the model's predictive power comes from. Raw data alone is rarely enough. You construct signals that capture the behavioural patterns most associated with churn.

The most important features fall into three categories. Recency signals tell you how recently the customer was active — days since last top-up, days since last data session. Frequency signals tell you how consistently they engage — number of top-ups in the last 30, 60, and 90 days, number of active days per month. Trend signals tell you whether behaviour is improving or deteriorating — data usage this month versus last month, top-up amount this month versus the 90-day average.

Trend features are particularly powerful because they capture momentum. A customer who topped up R50 last month but only R10 this month is sending a very different signal than a customer who consistently tops up R10. Both look the same in a snapshot but completely different in trend.

---

**Phase 4 — Data Preparation**

You clean the data, handle missing values, encode categorical variables into numbers, and split into training and test sets. The split must be stratified — meaning the churn rate in your training set and test set should mirror the churn rate in the full dataset. You never allow the model to see test data during training or preprocessing. Doing so is called data leakage and it produces falsely optimistic performance estimates that collapse in production.

You also address class imbalance here. In most telecom datasets, churners represent 15 to 30 percent of the base. If untreated, the model learns to predict the majority class and misses most actual churners. You handle this by telling the model to weight the minority class more heavily during training.

---

**Phase 5 — Model Development**

You build models in increasing order of complexity. You start with logistic regression as your baseline — it is fast, interpretable, and gives you a probability output immediately. You then move to a random forest which handles non-linear relationships and feature interactions that logistic regression cannot capture. You compare both using cross-validation so your performance estimates are reliable, not lucky.

For each model you are not just measuring overall accuracy. You are measuring how well it ranks customers — because in practice you are not treating everyone above a threshold equally. You are ranking the entire subscriber base by risk and working down the list until your intervention budget runs out.

---

**Phase 6 — Model Evaluation**

You evaluate against metrics that reflect the business problem, not just statistical performance. AUC-ROC is your primary metric — it measures how well the model separates churners from non-churners across all possible thresholds. Recall tells you what percentage of actual churners you caught. Precision tells you what percentage of your flagged customers were genuinely at risk. The confusion matrix shows exactly where the model succeeds and fails.

You then connect these numbers to business impact. If your model has 70% recall, that means you are catching 7 out of every 10 customers who would have churned. At a subscriber base of 1 million with a 10% churn rate, that is 70,000 customers identified for intervention. If your retention offer has a 40% success rate and each retained customer is worth R200 in the next 90 days, the recoverable revenue is R5.6 million from a single campaign cycle.

That is how you speak about model evaluation in this interview — not just numbers but what the numbers mean in rands and customers.

---

**Phase 7 — Threshold Optimisation**

The default decision threshold of 0.5 is rarely the right business choice. You analyse the precision-recall tradeoff across every possible threshold and choose the cutoff that aligns with the business's cost structure. If the retention offer costs R30 per customer and the average retained customer is worth R200, you can afford a relatively low precision — meaning you can tolerate some false positives. If the offer is a free device worth R1,500, you need high precision and should raise the threshold significantly.

This phase turns the model from a technical output into a business decision tool.

---

**Phase 8 — Segmentation Layer**

On top of the churn score you layer a value dimension. You do not treat all high-risk customers the same. A customer with a 0.80 churn probability who spends R500 a month is a completely different intervention priority from a customer with a 0.80 churn probability who spends R30 a month. You create a two-dimensional matrix — risk on one axis, value on the other — and this matrix drives differentiated treatment. High risk high value gets your best retention offer immediately. High risk low value gets a low-cost nudge or nothing at all.

This is exactly what your letter described as eligibility logic and behavioural tiers. The churn model feeds it directly.

---

**Phase 9 — Measuring Incremental Impact**

After deployment you measure whether the model actually worked. You do this by withholding a random sample of high-risk customers from the intervention — the holdout control group. After 90 days you compare churn rates between the treated group and the control group. The difference is your true incremental retention lift. This is the only honest way to prove that the model created business value rather than just identifying customers who would have stayed anyway.

Without this step you have a model. With this step you have a proven commercial asset.

---

**The One Paragraph That Ties It All Together**

If they ask you to summarise the entire approach in one minute, say this.

"The churn model takes every active prepaid subscriber, scores them daily on likelihood to become inactive within 90 days, ranks them by that score combined with their value to the business, and feeds a targeting engine that determines who gets which retention intervention at what cost. The model is evaluated not just on AUC-ROC but on how much incremental revenue it protects — measured against a holdout control group to isolate true causal impact from correlation. The full pipeline runs from raw CDR behavioural data through feature engineering, model training, threshold optimisation, and value-based segmentation, and it is rebuilt and recalibrated regularly as subscriber behaviour evolves."

---

That is your churn model. Own every word of it.