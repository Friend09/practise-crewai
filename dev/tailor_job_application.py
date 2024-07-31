from crewai import Agent, Task, Crew
import warnings
from langchain_openai import ChatOpenAI
from crewai_tools import FileReadTool, ScrapeWebsiteTool, MDXSearchTool, SerperDevTool
import os
from dotenv import load_dotenv, find_dotenv
from IPython.display import Markdown, display

def tailor_job_application(
    JOB_PROFILE: str,
    job_posting_url: str,
    github_url: str,
    personal_writeup: str,
    resume_md_path: str,
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
    - resume_md_path (str): The path to the Markdown file of the original resume.
    - output_resume_file (str): The file path where the tailored resume will be saved.
    - output_interview_file (str): The file path where the interview preparation materials will be saved.

    Returns:
    - None: The function saves the tailored resume and interview preparation materials to the specified file paths.
    """
    # Load environment variables
    _ = load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")

    # Initialize LLM
    # llm = ChatOpenAI(model="GPT-3.5-turbo")
    llm = ChatOpenAI(model="gpt-4")
    # llm = ChatOpenAI(model="gpt-4o")

    # Define tools
    tool_search = SerperDevTool()
    tool_scrape = ScrapeWebsiteTool()
    tool_read_resume = FileReadTool(file_path=resume_md_path)
    tool_semantic_search_resume = MDXSearchTool(mdx=resume_md_path)

    # Define agents
    agent_researcher = Agent(
        role="Tech Job Researcher",
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

    # Display results
    display(Markdown(output_resume_file))
    display(Markdown(output_interview_file))

# Example usage
# Define the job profile first
JOB_PROFILE = "Senior Application Developer AI"

tailor_job_application(
    JOB_PROFILE=JOB_PROFILE,
    job_posting_url="https://careers.marshmclennan.com/global/en/job/R_276678/Senior-Application-Developer-AI",
    github_url="https://github.com/Friend09",
    personal_writeup="As an experienced AI/ML Engineer, Raghu specialize in leveraging data science and machine learning to drive business transformation and operational excellence. His career is marked by a deep passion for developing innovative AI solutions that solve complex business challenges and enhance decision-making processes.",
    resume_md_path="/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_Interviews/subproj_Resume/files_proj_Resume/Raghu/Resume/raghuvamsi_ayapilla_resume_2024.md",
    output_resume_file=f"/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_AI/dev_proj_AI/pract-crewai/files/resume_{JOB_PROFILE}.md",
    output_interview_file=f"/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_AI/dev_proj_AI/pract-crewai/files/interview_{JOB_PROFILE}.md"
)
