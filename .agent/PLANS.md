# ExecPlan Specification  
This document defines the structure and usage rules for ExecPlans. An ExecPlan is a living, self-contained design document that guides complex or multi-hour development tasks. It allows an agent (human or AI) with no prior context to read the plan and execute all steps to produce a working outcome.
# ExecPlan Specification  
This document defines the structure and usage rules for ExecPlans. An ExecPlan is a living, self-contained design document that guides complex or multi-hour development tasks. It allows an agent (human or AI) with no prior context to read the plan and execute all steps to produce a working outcome.

## Purpose / Big Picture  
Explain in a few sentences what someone gains after this change and how they can see it working. State the user-visible behavior you will enable.  

## Progress  
Use a list with checkboxes to summarize granular steps. Every stopping point must be documented here, even if it requires splitting a partially completed task into two ("done" vs. "remaining"). This section must always reflect the current state of the work. Use timestamps to measure rates of progress.

## Surprises & Discoveries  
Capture observations about optimizer behavior, performance trade‑offs, unexpected bugs, or inverse/unapply semantics that shape the approach. When prototyping or exploring multiple implementations, log what you learn.

## Decision Log  
Record any decision points or forks in the plan. Document why a course was chosen and what alternatives were considered or discarded. Keep a history of changes in direction.

## Outcomes & Retrospective  
At completion of a major task or the full plan, write an entry summarizing what was achieved, what remains, and lessons learned. Include retrospective insights to inform future work.

## Context and Orientation  
Provide background on the problem, constraints, and any prior art. Reference any relevant standards or documents. Explain how this plan fits into the wider system.  

## Plan of Work  
Outline the phases or milestones of the work at a high level. Describe the order in which tasks will be tackled.  

## Concrete Steps  
List the specific steps required to complete the work. These should be detailed enough for a novice to follow without additional context.  

## Validation and Acceptance  
Describe how to test and verify the completed work. Define criteria for acceptance and any automated tests that should pass.  

## Idempotence and Recovery  
Explain how to restart or resume work if interrupted. The plan should allow repeatable execution without harmful side effects.  

## Artifacts and Notes  
List any output artifacts produced by following this plan (e.g., code files, documents). Add any supplementary notes or resources.

## Interfaces and Dependencies  
Document external interfaces, APIs, or dependencies that this work relies on. Specify version constraints or integration notes.
