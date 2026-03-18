# Supply Chain Copilot

A Streamlit-based Supply Chain Copilot that uses OpenAI plus Python tools to answer business questions from supply chain data.

## Live Demo

Recruiters and reviewers can view the deployed app here:

[https://baselstrasesintelligence.streamlit.app/](https://baselstrasesintelligence.streamlit.app/)

## Overview

This project demonstrates how agentic AI can support supply chain decision-making by combining:

- **Streamlit** for the user interface
- **OpenAI** for intent understanding and business-friendly responses
- **Python tools** for data retrieval and KPI analysis
- **CSV sample datasets** to simulate real supply chain operations

The goal is to help business users ask natural-language questions such as:

- Why is OTIF down?
- Which suppliers are causing delays?
- What is happening with forecast accuracy?
- Which materials are at inventory risk?

Instead of manually digging through SAP extracts and spreadsheets, the copilot routes the question to the right logic and returns a structured answer.

## Business Value

Supply chain teams often receive narrative questions from finance, operations, and leadership, for example:

- Why did service drop this month?
- What is driving stock risk?
- Which supplier issues are impacting performance?

This project shows how modern AI orchestration can turn fragmented data into faster root-cause visibility and better decision support.

## Features

- Natural-language question input
- Intent detection using OpenAI
- Tool-based routing for specific supply chain topics
- KPI calculations for OTIF, forecast error, inventory, and suppliers
- Business-style responses for decision-makers
- Sample CSV data for demonstration

## Project Structure

```text
supply_chain_copilot/
|-- agents/
|   `-- orchestrator.py
|-- data/
|   |-- forecast.csv
|   |-- inventory.csv
|   |-- otif.csv
|   |-- purchase_orders.csv
|   `-- suppliers.csv
|-- pages/
|   `-- 1_Copilot.py
|-- prompts/
|   `-- system_prompt.txt
|-- services/
|   `-- openai_client.py
|-- tools/
|   |-- forecast_tool.py
|   |-- inventory_tool.py
|   |-- otif_tool.py
|   `-- supplier_tool.py
|-- utils/
|   `-- metrics.py
|-- .env.example
|-- app.py
|-- requirements.txt
`-- README.md
```
