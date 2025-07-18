# Banking Campaign EDA

This is a data exploration project I worked on as part of a Data Science Experience Program. The dataset comes from a Portuguese bank’s marketing campaigns, which involved reaching out to clients through phone calls to promote term deposits.

My main goal was to understand how different client attributes and contact strategies influenced whether someone subscribed or not. I focused on exploring, visualizing, and interpreting — not building predictive models here.

---

## Dataset Overview

The dataset has 45,211 entries and 18 columns. Each row represents a client and their campaign interaction history. Some key columns include:

- Age, Job, Marital status, Education
- Financial info: balance, housing/personal loan status, credit in default
- Campaign details: contact method, day/month of contact, call duration, previous contacts
- Target variable: `y` (whether they subscribed to the term deposit)

---

## What I Did

- Broke down trends by **age, job, marital status, and education**
- Analyzed **client balance levels, overdrafts, and financial behavior**
- Compared **housing and personal loan** holders in terms of subscription
- Studied how **contact method (mobile/landline)** impacted conversion
- Looked at **call duration**, **number of contacts**, **recency**, and **previous campaign outcomes**
- Created multiple **distribution plots, boxplots, bar charts, and pie charts**
- Summarized insights for each section in my PPT

---

## Highlights & Learnings

- **Clients aged 60+** had the highest subscription rate (>40%)
- **Students** had the best job-based conversion (28.7%)
- **Clients without housing loans** were 2.2× more likely to subscribe
- Conversion was best when clients were contacted **2–3 times**, and within **1–3 months** of the last contact
- Call duration matters: most successful calls fell between 3–5 minutes
- **Mobile calls** dominated outreach but **landlines** performed better in terms of conversion
- I noticed some counterintuitive patterns (e.g., mid-aged clients had lower rates), which were interesting to dig into

---

## Files in This Repo

| File | Description |
|------|-------------|
| `Project-1_Banking.py` | Full analysis code — includes data cleaning, EDA, plots |
| `Problem Statement & Data Description.docx` | The original case prompt |
| `Case Project_PPT.pptx` | Slide deck summarizing my findings visually |

---

## Tools & Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn

---

## About Me

I'm Aryan Aher, a student at IIT Kharagpur.  
This was one of the first end-to-end EDA projects where I got to apply code and logic to real-world behavioral data.  
I learned a lot about storytelling with data, especially how patterns can reveal the “why” behind client behavior.

📫 aryanraher@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/your-profile](https://www.linkedin.com/in/aryanraher/ )

---

Thanks for stopping by. If you have feedback or suggestions, feel free to reach out.
