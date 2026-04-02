## Reviewer #4 Comments

This paper proposes the **LAD-Flex** strategy, which filters loads that can be postponed using task queuing latency, and implements short-term load reduction and peak shifting at the request of aggregators. However, several aspects require further improvement:

1. **Oversimplified GPU power model**  
   The assumption of a linear relationship between GPU power and load is oversimplified, as modern GPUs typically exhibit highly non-linear power curves. Please discuss how this simplification impacts your findings.  
   **Reply to reviewer:**  
   We thank the reviewer for this important comment. We agree that modern GPUs exhibit nonlinear power-utilization characteristics and that a linear relationship is a simplification. In this study, however, the Alibaba trace does not provide measured per-device power telemetry or a reliable mapping from each task to the utilization trajectory of an individual physical GPU over time. The trace identifies the GPU model/group associated with a task together with its allocated GPU fraction, but not the machine-level operating point needed to calibrate device-specific nonlinear power curves. For this reason, we retained a first-order linear GPU power model and clarified that the objective of the paper is workload-level flexibility assessment, rather than hardware-accurate GPU power characterization. We have revised the manuscript to explain this modeling choice explicitly and to discuss its implications for the interpretation of the results.  
   **Action taken:**  
   We revised the methodology and case-study sections to clarify that GPU power is estimated from GPU model/group information and allocated utilization fractions because the dataset does not provide per-device telemetry or reliable task-to-individual-GPU utilization traces. We also added a limitation paragraph stating that the adopted linear model is a first-order approximation of GPU-side IT power, that absolute power and energy magnitudes may therefore be misestimated, especially for fractional-GPU allocations and nonlinear operating regions, and that the results are more robust when interpreted in relative terms for flexibility assessment. Finally, we softened the interpretation of the reported power values and added future-work text indicating that nonlinear node/GPU power models should be calibrated using measured telemetry or machine-level monitoring data when such data become available.  

2. **Discussion of PUE and non-IT power**  
   Since non-IT power accounts for a significant portion of data center energy use, it would be valuable to discuss the implications of the proposed method in terms of **Power Usage Effectiveness (PUE)**.  
   **Reply to reviewer:**  
   We thank the reviewer for this valuable comment. We agree that non-IT loads and PUE are important when interpreting demand response potential at the whole-facility level. In the present study, however, LAD-Flex acts directly on IT workloads, and the Alibaba trace does not provide facility-level telemetry, cooling-system measurements, or dynamic PUE data. As a result, the current methodology cannot quantify whole-site response or short-term PUE dynamics empirically. We therefore clarified in the manuscript that the reported results should be interpreted as IT-side flexibility estimates, while the facility-level impact depends on how cooling and auxiliary loads respond during the activation window.  
   **Action taken:**  
   We revised the introduction, methodology, results interpretation, and limitations sections to distinguish IT-side flexibility from whole-facility flexibility more explicitly. We now state that non-IT systems such as cooling, UPS, and auxiliaries are outside the modeled scope of the present analysis, that dynamic PUE effects cannot be quantified with the available trace, and that whole-site reduction should not be inferred directly from the IT-side results without additional assumptions. We also added a limitation paragraph explaining that if non-IT loads do not decrease proportionally during short flexibility events, instantaneous PUE may temporarily worsen, and that future work should combine workload-level flexibility assessment with facility telemetry or thermal/infrastructure models to assess whole-site response and short-term PUE dynamics.  

3. **Workflow dependencies in the Alibaba dataset**  
   In the Alibaba dataset, many tasks within the same group may belong to a single workflow, which is not uncommon. The authors should clarify whether delaying one task could trigger cascading effects on subsequent tasks and, consequently, degrade overall **QoS**.  
   **Reply to reviewer:**  
   **Action taken:**  

4. **Lack of empirical support for \(\alpha\) and \(\tau\)**  
   The parameters \(\alpha\) and \(\tau\) lack empirical grounding in real-world electricity markets. In particular, the assumption \(V_{max} = €2/\text{kWh}\) appears excessively high compared to average bids observed in most European ancillary service markets.  
   **Reply to reviewer:**  
   **Action taken:**  

5. **Quantification of the rebound effect**  
   The authors are encouraged to quantify the impact of the **rebound effect** in order to validate the claimed efficiency gains.  
   **Reply to reviewer:**  
   **Action taken:**  

6. **Confusing time axis in the figure**  
   The use of “1970” in the figure is confusing, even though the authors clarify that it refers to the Unix epoch. The axis should be changed to **relative time** for better readability.  
   **Reply to reviewer:**  
   **Action taken:**  