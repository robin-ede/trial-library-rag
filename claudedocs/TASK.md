Project Overview
You'll be building a question-answering tool from scratch that can find answers within a collection of PDF documents. This assessment is your chance to showcase your ability to work with LLMs, design data preparation pipelines, implement retrieval systems, and evaluate your work—all skills that are central to this role.

We normally give up to a week to complete your app. But since it is a holiday, I will set the due date at EOD December 1st. Please turn in your finished submission via a link to a public GitHub repository.
Technical Requirements
Desired Stack
Python (3.9+)
LLM Provider: OpenAI API, Anthropic API, or any other LLM API of your choice
UI Framework: A terminal app is fine, or if you would like to build a simple GUI, Streamlit or Gradio would be sufficient (or similar—we want something functional, not polished or complex)
Data Selection
Choose 3-5 PDF documents to use as your knowledge base. These can be:

Technical documentation
Research papers
Reports
Books or book chapters
Any other PDF content that interests you

The goal is to have enough content to demonstrate your retrieval and data processing strategies, but not so much that data preparation becomes the entire project.
Feature Requirements
Your application should include:

Question-Answering System

The user should be able to ask questions about the content of the documents using natural language
The answers should include LLM generated text as well as basic citations

Basic User Interface

Simple UI for asking questions
Display answers with source attribution

Evaluation Framework

Create at least 5 test questions with expected answers
Implement at least 2 evaluation metrics (e.g., retrieval accuracy, answer relevance, etc.)
Feel free to use any contemporary method to assist you in establishing a ground truth for these metrics
Document your evaluation results
Discuss what's working well and what isn't

Error Handling and Edge Cases

We would like to see some error handling, but will leave it to you to decide what is important
What to Submit
A public GitHub repository containing your complete project linked in a reply to this email.
Repository Requirements
An included README.md file with:

Setup and installation instructions (including API key setup – don’t commit your credentials)
A brief description of your approach and design decisions
Which technologies were previously familiar vs. new to you
Evaluation results and your interpretation
Known limitations or issues
What you would add/remove/change with more time

All source code and configuration files

Your evaluation code

A requirements.txt or pyproject.toml file

Sample PDFs you used (or clear instructions on where to obtain them)

A .env.example file showing what environment variables are needed

A commit history showing your development process
Notes
API Costs: We understand API calls cost money. Feel free to use smaller models or limit the scope of your evaluation to manage costs. Model accuracy isn’t important to us – only the methodology used.

Creative Control: You have flexibility in your technical choices. We care more about your rationale and evaluation of those choices than the specific tools you pick.