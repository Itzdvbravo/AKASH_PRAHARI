# TerraEyes — Detailed Implementation Planning and Controlled Development

## 1. Project Context

I am developing **TerraEyes**, an application for semantic retrieval and multi-temporal change analysis of satellite imagery.

The four reference images provided above describe the application's architecture, workflow, and expected output. The fourth image represents the application's final summary/output interface. Study all four images carefully and use them as the primary reference for understanding the intended application.

The application is intended to support workflows such as:

* Searching satellite imagery using natural-language queries.
* Retrieving relevant locations and satellite images based on semantic similarity.
* Comparing satellite imagery across different time periods.
* Detecting and visualizing changes between two or more dates.
* Identifying changes such as vegetation loss, newly constructed buildings, and other land-cover transformations.
* Displaying relevant satellite imagery, change visualizations, bounding boxes, and a structured summary containing the location, time range, detected change, earliest supported change, and confidence.
* Supporting satellite imagery from Landsat, Sentinel-1, and Sentinel-2, according to the architecture.
* Providing an analyst-oriented interface that makes querying, comparing, and interpreting satellite imagery straightforward.

The project will initially use the **OSCD (Onera Satellite Change Detection) dataset** for experimentation and implementation. However, do not download the dataset, generate a large dataset copy, or begin dataset preprocessing at this stage. Only plan the required dataset-handling structure and integration points.

Treat the reference images as the intended design and architectural direction, but identify any missing technical details, dependencies, or assumptions that need to be resolved before implementation.

## 2. Immediate Objective: Planning Only

For this stage, **do not write application code, create project files, install dependencies, download datasets, initialize repositories, or implement any application components.**

Your only task is to produce an extensive, detailed, technically grounded implementation plan.

The plan must be sufficiently detailed that individual team members can independently implement different modules while following a shared architecture and integration contract.

Do not begin implementation until I explicitly instruct you to do so.

## 3. Development Strategy: Working Prototype First

The project must follow a prototype-first, incremental development strategy.

### Phase A: Foundational Working Prototype

Before attempting advanced models, establish a minimal but functional end-to-end application that satisfies the basic requirements.

The initial prototype should aim to provide:

1. A functioning frontend with the basic application layout and navigation.
2. A backend exposing the necessary API endpoints.
3. A clearly defined request/response contract between the frontend and backend.
4. A working query-submission and results-display workflow.
5. The ability to display satellite images and basic metadata using a small, controlled set of sample records or mock data.
6. A basic image-comparison view for two dates, using suitable sample imagery.
7. A simple baseline change-detection workflow, even if it initially uses a conventional method or a lightweight existing model.
8. A preliminary change visualization, such as a difference mask or overlay.
9. A basic result summary containing the location, dates, detected changes, and relevant confidence or uncertainty information.
10. Basic validation, error handling, loading states, and instructions for running the application locally.

The initial prototype does not need to implement every advanced capability described in the reference architecture.

However, it must be a genuinely working application rather than a collection of disconnected UI screens, placeholder endpoints, or architectural diagrams.

Use mock data only where necessary to unblock development. Clearly identify mocked functionality and ensure the architecture allows it to be replaced with actual dataset records, satellite imagery, and trained or pretrained models later.

Where an actual satellite image is needed, use a small, legally usable sample or a locally supplied image. Do not download OSCD at this stage.

**Prototype acceptance criterion:** A user should be able to submit a query or select a location and time range, receive a result through the backend, view imagery from two dates, inspect a basic change visualization, and see a structured result summary.

If any requirement cannot reasonably be achieved in the first prototype, identify it explicitly and propose a staged alternative.

### Phase B: Functional Expansion

After the prototype works, progressively introduce:

* OSCD dataset integration and preprocessing.
* Satellite image and metadata ingestion.
* Geospatial coordinate handling and image alignment.
* Semantic image retrieval.
* Multi-temporal image pairing.
* Model-based change detection.
* Bounding-box or polygon generation where appropriate.
* Confidence calibration and uncertainty handling.
* Support for Landsat, Sentinel-1, and Sentinel-2.
* Improved visualization, filtering, and analyst workflows.
* Performance optimization and offline execution where feasible.

Do not implement these features prematurely. Determine their dependencies and the order in which they should be introduced.

### Phase C: Validation and Optimization

Plan for:

* Evaluation of semantic retrieval quality.
* Change-detection evaluation using suitable metrics, such as precision, recall, F1-score, and IoU.
* Testing across different geographic areas and change types.
* Validation of image registration and temporal consistency.
* Model inference performance and memory requirements.
* Robustness to differences between satellite sensors.
* Testing with actual data rather than only curated examples.
* Comparison of baseline and advanced approaches.

Clearly distinguish prototype demonstrations from experimentally validated results.

## 4. Repository and Folder Architecture

Design a modular repository that allows team members to work independently without constantly modifying the same files.

Use three principal implementation areas:

* `frontend/` — user interface, application state, visualization, and API integration.
* `backend/` — API, business logic, orchestration, model integration, and result generation.
* `data-handling/` — dataset integration, satellite imagery ingestion, preprocessing, geospatial operations, metadata, and dataset-specific utilities.

Include a separate `setup/` area for foundational configuration, development tooling, environment setup, and shared project conventions.

You may propose additional root-level folders where justified, such as `docs/`, `tests/`, or `scripts/`, but keep the main responsibilities of the three implementation areas separate.

Do not create this structure yet. Specify it in the plan only.

### 4.1 Setup Area

Plan a foundational setup phase that defines:

* Repository initialization and branch conventions.
* Environment and dependency management.
* Configuration and environment-variable conventions.
* Frontend and backend development commands.
* API base URLs and development communication.
* Dependency and model version pinning.
* Shared coding standards and formatting.
* Logging and error-handling conventions.
* Test framework configuration.
* Local development and startup instructions.
* Configuration templates that do not contain secrets.
* Optional containerization if justified by the project requirements.

Explain which setup tasks must be completed before parallel development begins and which can be deferred.

Avoid prematurely introducing infrastructure that is unnecessary for a local prototype.

### 4.2 Frontend Structure

Propose a detailed, feature-oriented frontend folder structure.

Separate responsibilities into independently implementable modules, including:

* Application bootstrap and routing.
* Shared layout and navigation.
* Search and query input.
* Query parameters and validation.
* Location and time-range selection.
* Satellite image display.
* Temporal comparison interface.
* Change-mask and overlay visualization.
* Bounding-box or polygon visualization.
* Result summary panel.
* Metadata and confidence display.
* API client and endpoint-specific services.
* Application state management.
* Shared UI components.
* Type definitions and API response schemas.
* Error, loading, and empty states.
* Frontend tests.

For each major module, explain its responsibilities, expected inputs and outputs, important files, dependencies, and acceptance criteria.

Make it possible for different team members to work on the search interface, visualization components, and result-summary interface in parallel.

Do not overengineer the frontend. Select a reasonable architecture that fits the existing project and the initial prototype.

### 4.3 Backend Structure

Propose a detailed backend structure with clearly separated responsibilities.

Include modules for:

* Application initialization and configuration.
* API routing and versioning.
* Request validation and response schemas.
* Query interpretation and orchestration.
* Search and retrieval.
* Image and metadata retrieval.
* Temporal image-pair selection.
* Change-detection inference.
* Semantic embedding generation and similarity search.
* Geospatial processing and coordinate transformations.
* Change-mask postprocessing.
* Bounding-box or polygon generation, if required.
* Confidence and uncertainty reporting.
* Result aggregation and summary generation.
* Model loading, model interfaces, and inference utilities.
* Logging, exceptions, and error responses.
* Health checks.
* Unit and integration tests.

Explain which modules should be independent of specific machine-learning models and which should contain model-specific implementations.

Define service interfaces so that a baseline model can be replaced by a more advanced model without rewriting the API or frontend.

Clarify whether the initial backend should be a modular monolith rather than a collection of microservices, and justify the choice.

### 4.4 Data-Handling Structure

Design `data-handling/` as a separate area responsible for preparing and providing satellite data to the backend.

Plan modules for:

* Dataset adapters.
* OSCD-specific loading and parsing.
* Dataset directory conventions.
* Metadata extraction and normalization.
* Image discovery and pairing across dates.
* Image reading and format conversion.
* Band selection and normalization.
* Image registration and alignment.
* Geospatial metadata and coordinate reference systems.
* Resampling and resolution handling.
* No-data regions and missing imagery.
* Preprocessing pipelines.
* Train, validation, and test splits where relevant.
* Dataset integrity checks.
* Small-sample fixtures and synthetic test records.
* Dataset documentation and provenance.

Keep dataset-specific logic isolated from generic preprocessing utilities.

The backend should consume a stable interface from this layer rather than directly depending on OSCD's directory structure or hardcoded file paths.

Explain how the data-handling layer will eventually accommodate additional sources, particularly Landsat, Sentinel-1, and Sentinel-2.

**Important:** OSCD has not been downloaded. Do not assume its files, bands, metadata, or directory structure have already been inspected. Identify what must be verified when the dataset becomes available.

## 5. Model Selection and Technical Decisions

Do not prematurely lock the project into a specific model or model architecture.

Create a model-selection section that identifies candidate approaches for each major capability.

### 5.1 Semantic Retrieval

Evaluate suitable approaches for matching natural-language queries with satellite imagery.

Consider relevant remote-sensing vision-language models, including RemoteCLIP, alongside suitable alternatives.

Discuss:

* Compatibility with satellite imagery.
* Semantic retrieval quality.
* Model size and inference requirements.
* Embedding generation and storage.
* Suitability for offline use.
* Whether image-level embeddings are sufficient or metadata and geospatial filtering are also required.
* Limitations of retrieving specific changes from a single image.

### 5.2 Change Detection

Evaluate candidate approaches for identifying changes between images acquired at different times.

Consider:

* A simple image-difference baseline.
* Conventional image-processing methods.
* A pretrained or existing change-detection model.
* A suitable learned model that can be evaluated on OSCD.
* More advanced approaches only if justified by the available time and hardware.

Discuss registration errors, seasonal variation, illumination differences, sensor differences, and false positives.

Do not assume that a semantic retrieval model can perform pixel-level change detection merely because it can compare image embeddings.

### 5.3 Satellite-Specific Processing

Explain the implications of supporting:

* Landsat.
* Sentinel-1 SAR imagery.
* Sentinel-2 optical imagery.

Identify the differences in bands, spatial resolution, preprocessing, and data characteristics that affect model compatibility.

Do not assume one model can directly process all three sources without appropriate preprocessing, modality handling, or validation.

### 5.4 Model Feedback and Approval Gates

At relevant points in the implementation plan, include explicit model-selection checkpoints.

For each checkpoint, specify:

1. The technical decision to be made.
2. The viable alternatives.
3. The evidence or experiment needed.
4. The computational and implementation costs.
5. The expected benefits and limitations.
6. The decision I need to approve before proceeding.

When the implementation reaches one of these checkpoints, stop and present the relevant options with a reasoned comparison. Ask for my feedback before committing to a consequential model or architecture change.

Do not repeatedly ask for approval of minor implementation details that can be resolved within the agreed architecture.

## 6. API Contracts and Integration

Define the expected API contracts before assigning parallel implementation work.

Propose endpoints for the prototype and identify additional endpoints that may be introduced later.

Potential operations include:

* Application health checks.
* Query submission.
* Search result retrieval.
* Satellite image and metadata retrieval.
* Temporal comparison requests.
* Change-detection execution.
* Change visualization and result-summary retrieval.

For each endpoint, specify:

* HTTP method and path.
* Request schema.
* Response schema.
* Required and optional fields.
* Validation rules.
* Error responses.
* Whether the endpoint belongs to the prototype or a later phase.

Define consistent identifiers, timestamp formats, coordinate conventions, image references, and confidence-field semantics.

The frontend and backend must be able to develop against agreed schemas and mock responses before the complete data pipeline is available.

Avoid duplicating model logic in the frontend.

## 7. Parallel Team Development

Assume that multiple team members will work on the project simultaneously.

Organize the plan into workstreams that can proceed independently after the foundational setup and interface contracts are established.

For every workstream, provide:

* Scope and responsibilities.
* Required knowledge or skills.
* Prerequisites.
* Exact modules or folders owned by the workstream.
* Expected deliverables.
* Inputs and outputs.
* Dependencies on other workstreams.
* Integration points.
* Acceptance criteria.
* Tests required before integration.

Consider separate workstreams for:

1. Setup, repository conventions, and shared contracts.
2. Frontend search and application layout.
3. Frontend temporal visualization and result summary.
4. Backend API and orchestration.
5. Data-handling abstractions and OSCD integration.
6. Semantic retrieval and change-detection model evaluation.

These are proposed workstreams, not a requirement to assign one person to every item. Recommend a practical division based on team size.

Make it explicit which files or interfaces each workstream owns. Minimize overlapping ownership and shared-file conflicts.

Use mock implementations and agreed interfaces to unblock parallel development where appropriate.

## 8. Dependency Graph and Implementation Sequence

Create a step-by-step implementation roadmap.

For each step, specify:

* Objective.
* Tasks in execution order.
* Files or modules expected to be involved.
* Dependencies.
* Deliverables.
* Verification steps.
* Acceptance criteria.
* Whether work can proceed in parallel.
* Whether my approval is required before moving forward.

Start with setup and a foundational working prototype. Introduce real dataset integration and advanced model capabilities only when their prerequisites are satisfied.

Show the dependency graph between setup, frontend, backend, data handling, model evaluation, and integration.

Clearly identify critical-path tasks and opportunities for parallel work.

## 9. Testing and Quality Assurance

Include a testing strategy covering:

* Unit tests for independent modules.
* API contract tests.
* Frontend component tests.
* Integration tests.
* Data-loading and preprocessing tests.
* Image-pair validation.
* Model inference tests.
* End-to-end prototype tests.
* Regression tests when models or preprocessing change.

Define measurable acceptance criteria wherever practical.

For model evaluation, distinguish between retrieval metrics and change-detection metrics. Do not use an arbitrary confidence score as evidence of model accuracy.

Include checks to ensure that timestamps, locations, imagery, and detected changes remain consistent throughout the workflow.

## 10. Constraints and Design Principles

Follow these principles throughout the plan:

* Prototype first, advanced features later.
* Modular architecture with clear boundaries.
* Independent implementation of frontend, backend, and data handling.
* Stable API contracts.
* Replaceable model implementations.
* No premature microservices or unnecessary infrastructure.
* No downloading OSCD during the planning stage.
* Avoid unnecessary duplication of data and model artifacts.
* Support local development and consider offline operation in the architecture.
* Do not fabricate model performance, dataset properties, or experimental results.
* Identify assumptions and unresolved questions explicitly.
* Keep the initial implementation achievable within a limited development timeline.

Where a requirement is too expensive for the initial prototype, propose a simpler alternative without silently removing the requirement from the overall roadmap.

## 11. Required Deliverable: The Implementation Plan

Return a detailed plan with the following sections:

1. Understanding of the reference architecture and application requirements.
2. Explicit assumptions, ambiguities, and questions requiring clarification.
3. Proposed technology stack and justification.
4. Repository architecture and complete proposed folder tree.
5. Responsibility of every major folder and module.
6. Foundational setup plan.
7. Foundational working prototype plan.
8. Frontend implementation plan.
9. Backend implementation plan.
10. Data-handling and OSCD integration plan.
11. Model candidates and model-selection checkpoints.
12. API contracts and shared schemas.
13. Team workstreams and ownership boundaries.
14. Dependency graph and parallel execution opportunities.
15. Detailed, ordered implementation roadmap.
16. Testing strategy and acceptance criteria.
17. Risks, limitations, and mitigation strategies.
18. Final checklist for determining when each phase is complete.

Be specific about modules, interfaces, dependencies, and deliverables. Avoid generic advice such as "implement the backend" without explaining the actual work involved.

## 12. Strict Execution Control

This is a planning and approval-driven development process.

**After presenting the implementation plan, stop. Do not start coding or create any files.**

Once I approve the plan, implement only the first explicitly authorized step.

For every implementation step:

1. State the objective and scope.
2. Identify the files and modules to be created or modified.
3. Identify dependencies and assumptions.
4. Implement only the authorized scope.
5. Run the relevant tests or verification commands.
6. Summarize the changes and test results.
7. Identify incomplete work, known issues, and the next logical step.
8. Stop and wait for my instruction before proceeding.

Do not automatically continue into the next phase, implement unrelated improvements, or make consequential model-selection decisions without approval.

If you discover that the existing architecture or implementation plan requires a significant change, explain the reason, present the alternatives, and request my approval before proceeding.

The goal is to build TerraEyes incrementally, maintain control over architectural decisions, enable parallel team development, and establish a working prototype before investing in advanced functionality.
