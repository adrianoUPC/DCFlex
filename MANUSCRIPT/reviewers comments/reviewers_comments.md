## Reviewer #4 Comments

This paper proposes the **LAD-Flex** strategy, which filters loads that can be postponed using task queuing latency, and implements short-term load reduction and peak shifting at the request of aggregators. However, several aspects require further improvement:

1. **Oversimplified GPU power model**  
   The assumption of a linear relationship between GPU power and load is oversimplified, as modern GPUs typically exhibit highly non-linear power curves. The authors should discuss how this simplification affects the validity of their findings.

2. **Discussion of PUE and non-IT power**  
   Since non-IT power accounts for a significant portion of data center energy use, it would be valuable to discuss the implications of the proposed method in terms of **Power Usage Effectiveness (PUE)**.

3. **Workflow dependencies in the Alibaba dataset**  
   In the Alibaba dataset, many tasks within the same group may belong to a single workflow, which is not uncommon. The authors should clarify whether delaying one task could trigger cascading effects on subsequent tasks and, consequently, degrade overall **QoS**.

4. **Lack of empirical support for \(\alpha\) and \(\tau\)**  
   The parameters \(\alpha\) and \(\tau\) lack empirical grounding in real-world electricity markets. In particular, the assumption \(V_{max} = €2/\text{kWh}\) appears excessively high compared to average bids observed in most European ancillary service markets.

5. **Quantification of the rebound effect**  
   The authors are encouraged to quantify the impact of the **rebound effect** in order to validate the claimed efficiency gains.

6. **Confusing time axis in the figure**  
   The use of “1970” in the figure is confusing, even though the authors clarify that it refers to the Unix epoch. The axis should be changed to **relative time** for better readability.