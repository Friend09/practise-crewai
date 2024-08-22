from crewai import Agent, Task, Crew
import warnings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from crewai_tools import FileReadTool, ScrapeWebsiteTool, MDXSearchTool, SerperDevTool
import os
from dotenv import load_dotenv, find_dotenv
from IPython.display import Markdown, display

# This is my professional career profile (just keep updating this)
PROFESSIONAL_PROFILE_MD="/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_Interviews/subproj_Resume/files_proj_Resume/Raghu/Resume/raghuvamsi_ayapilla_resume_details_for_ai_2024.md"

def tailor_job_application(
    JOB_PROFILE: str,
    job_posting_url: str,
    github_url: str,
    personal_writeup: str,
    output_resume_file: str,
    output_interview_file: str
):
    """
    Tailor a job application by analyzing a job posting, compiling a personal profile,
    refining a resume, and preparing for interviews.

    Parameters:
    - JOB_PROFILE (str): The job profile title for the application (e.g., 'Senior Application Developer AI').
    - job_posting_url (str): URL of the job posting to analyze.
    - github_url (str): URL of the GitHub profile to use for extracting personal project information.
    - personal_writeup (str): A write-up providing a personal summary or introduction.
    - output_resume_file (str): The file path where the tailored resume will be saved.
    - output_interview_file (str): The file path where the interview preparation materials will be saved.

    Returns:
    - None: The function saves the tailored resume and interview preparation materials to the specified file paths.
    """
    # Load environment variables
    _ = load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    CLAUDEAI_API_KEY = os.getenv("CLAUDEAI_API_KEY")

    # Initialize LLM
    # llm = ChatOpenAI(model="GPT-3.5-turbo")
    llm = ChatOpenAI(model="gpt-4-turbo")
    # llm = ChatOpenAI(model="gpt-4o")
    # llm = ChatAnthropic(model="claude-3-opus-20240229")

    # Define tools
    tool_search = SerperDevTool()
    tool_scrape = ScrapeWebsiteTool()
    tool_read_resume = FileReadTool(file_path=PROFESSIONAL_PROFILE_MD)
    tool_semantic_search_resume = MDXSearchTool(mdx=PROFESSIONAL_PROFILE_MD)

    # Define agents
    agent_researcher = Agent(
        role="Tech Job Researcher", # dont change this. this is the agent ROLE
        goal="Make sure to do amazing analysis on job posting to help job applicants",
        backstory="As a Job Researcher, your prowess in navigating and extracting critical information from job postings is unmatched. Your skills help pinpoint the necessary qualifications and skills sought by employers, forming the foundation for effective application tailoring.",
        tools=[tool_scrape, tool_search],
        llm=llm,
        verbose=True,
    )

    agent_profiler = Agent(
        role=f"Personal Profiler for {JOB_PROFILE}s",
        goal="Do incredible research on job applicants to help them stand out in the job market",
        backstory="Equipped with analytical prowess, you dissect and synthesize information from diverse sources to craft comprehensive personal and professional profiles, laying the groundwork for personalized resume enhancements.",
        tools=[tool_scrape, tool_search, tool_read_resume, tool_semantic_search_resume],
        llm=llm,
        verbose=True,
    )

    agent_resume_strategist = Agent(
        role=f"Resume Strategist for {JOB_PROFILE}s",
        goal=f"Find all best ways to make an {JOB_PROFILE}'s resume more professional and stand out in the current job market.",
        backstory="With a strategic mind and an eye for detail, you excel at refining resumes to highlight the most relevant skills and experiences, ensuring they resonate perfectly with the job's requirements.",
        tools=[tool_scrape, tool_search, tool_read_resume, tool_semantic_search_resume],
        llm=llm,
        verbose=True,
    )

    agent_interview_preparer = Agent(
        role=f"{JOB_PROFILE}'s interview preparer",
        goal=f"Create interview questions and talking points based on the resume and job requirements.",
        backstory="Your role is crucial in anticipating the dynamics of interviews. With your ability to formulate key questions and talking points, you prepare candidates for success, ensuring they can confidently address all aspects of the job they are applying for.",
        tools=[tool_scrape, tool_search, tool_read_resume, tool_semantic_search_resume],
        llm=llm,
        verbose=True,
    )

    # Define tasks
    task_research = Task(
        description=f"Analyze the job posting URL provided ({job_posting_url}) to extract key skills, experiences, and qualifications required. Use the tools to gather content and identify and categorize the requirements.",
        expected_output="A structured list of job requirements, including necessary skills, qualifications, and experiences.",
        agent=agent_researcher,
        async_execution=True,
    )

    task_profile = Task(
        description=f"Compile a detailed personal and professional profile using the GitHub ({github_url}) URLs, and personal write-up ({personal_writeup}). Utilize tools to extract and synthesize information from these sources.",
        expected_output="A comprehensive profile document that includes skills, project experiences, contributions, interests, and communication style.",
        agent=agent_profiler,
        async_execution=True,
    )

    task_resume_strategy = Task(
        description=f"Using the profile and job requirements obtained from previous tasks, tailor the resume to highlight the most relevant areas. Employ tools to adjust and enhance the resume content. Make sure this is the best resume even BUT DO NOT make up any information. Update every section, including the initial summary, work experience, skills, and education. All to better reflect the candidate's abilities and how it matches the job posting.",
        expected_output="An updated resume that effectively highlights the candidate's abilities and experiences relevant to the job.",
        output_file=output_resume_file,
        context=[task_research, task_profile],
        agent=agent_resume_strategist,
    )

    task_interview_preparation = Task(
        description=f"Create a set of potential interview questions and talking points based on the tailored resume and job requirements. Utilize tools to generate relevant questions and discussion points. Make sure to use these questions and talking points to help the candidate highlight the main points of the resume and how it matches the job posting.",
        expected_output="A document containing key questions and talking points that the candidate should prepare for the initial interview.",
        output_file=output_interview_file,
        context=[task_research, task_profile, task_resume_strategy],
        agent=agent_interview_preparer,
    )

    # Define crew
    crew_job_application = Crew(
        agents=[
            agent_researcher,
            agent_profiler,
            agent_resume_strategist,
            agent_interview_preparer,
        ],
        tasks=[
            task_research,
            task_profile,
            task_resume_strategy,
            task_interview_preparation,
        ],
        manager_llm=llm,
        verbose=True,
    )

    # Inputs for the crew
    inputs_job_application = {
        "JOB_PROFILE": JOB_PROFILE,
        "job_posting_url": job_posting_url,
        "github_url": github_url,
        "personal_writeup": personal_writeup,
    }

    # Kickoff the crew
    result = crew_job_application.kickoff(inputs=inputs_job_application)

    # confirmation of end of script
    print("Script Ended! please check path at: `/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_AI/dev_proj_AI/pract-crewai/files` for generated files")

# Example usage
# Define the job profile first
JOB_PROFILE = "" # August 22, 2024 -> Lead Data Scientist

tailor_job_application(
    JOB_PROFILE=JOB_PROFILE,
    job_posting_url="https://careers.marshmclennan.com/global/en/job/R_276689/Lead-Data-Scientist",
    github_url="https://github.com/Friend09",
    personal_writeup="""
As a seasoned AI/ML Engineer with over 10 years of experience in Agile Project Management, 4+ years in data science, and 10 years in business systems analysis, I am excited to bring my extensive expertise to the Lead Data Scientist role at Marsh McLennan. My career has been defined by a commitment to delivering transformative data solutions, analyzing complex data using statistical and machine learning models, and driving innovation in IT operations through advanced analytics.

Throughout my tenure at Marsh McLennan, I have successfully identified business needs and translated them into actionable data-driven insights and machine learning models that drive value across various business lines. My proficiency in Python for data analysis, visualization, and working with APIs has been instrumental in developing innovative solutions for natural language processing and generative modeling tasks using NLP, Generative AI, and LLMs.

A key project that showcases my capabilities is the end-to-end tag prediction AI model I developed for ServiceNow incidents. This project involved extracting data from ServiceNow using its REST API, preprocessing the data, and implementing a tag prediction model using Langchain and OpenAI. The success of this initiative significantly improved categorization accuracy and enhanced planning and reporting capabilities for our teams.

My experience with cloud technologies, particularly AWS, coupled with my proficiency in Agile Scrum methodologies, has enabled me to guide data science teams through organizational challenges and deliver automation and intelligent recommendations that improve analytical outcomes. I have a proven track record of leading cross-functional data initiatives from planning through to deployment, ensuring successful implementation of machine learning models and continuous improvement of data products.

With a strong foundation in machine learning, AI-driven solutions, and MLOps practices, I have implemented various projects that have significantly improved IT operations and service management. My work on AI-based projects, such as developing chatbots for mean time to resolution analysis and automating ticket tagging using AI-based model predictions, demonstrates my ability to deliver impactful results through advanced data science techniques.

I am passionate about collaborating with cross-functional teams, understanding complex business requirements, and translating them into effective data science solutions. My experience in business analysis and working within an Agile framework has honed my ability to communicate findings and insights effectively to both technical and non-technical stakeholders.

As a Lead Data Scientist at Marsh McLennan, I am eager to leverage my technical expertise and leadership skills to drive innovative data science initiatives, mentor team members, and contribute to the company's mission of solving complex challenges through data-driven innovation. I am confident that my combination of technical proficiency, business acumen, and collaborative approach will make a significant impact on the data science team and the broader Marsh McLennan organization.
""",
    output_resume_file=f"/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_AI/dev_proj_AI/pract-crewai/files/resume_{JOB_PROFILE}.md",
    output_interview_file=f"/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_AI/dev_proj_AI/pract-crewai/files/interview_{JOB_PROFILE}.md"
)
