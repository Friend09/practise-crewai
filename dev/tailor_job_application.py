from crewai import Agent, Task, Crew
import warnings
from langchain_openai import ChatOpenAI
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
    llm = ChatOpenAI(model="gpt-4")
    # llm = ChatOpenAI(model="gpt-4o")

    # Define tools
    tool_search = SerperDevTool()
    tool_scrape = ScrapeWebsiteTool()
    tool_read_resume = FileReadTool(file_path=PROFESSIONAL_PROFILE_MD)
    tool_semantic_search_resume = MDXSearchTool(mdx=PROFESSIONAL_PROFILE_MD)

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
JOB_PROFILE = "Sr-Manager-Technical-Product-Manager"

tailor_job_application(
    JOB_PROFILE=JOB_PROFILE,
    job_posting_url="https://cvshealth.wd1.myworkdayjobs.com/CVS_Health_Careers/job/IL---Buffalo-Grove/Sr-Manager--Technical-Product-Manager_R0260972?shared_id=YTEwOTQ4OWItMGMyNS00MGY4LWE5MjAtZWI1MjI3ZTY0NTVl",
    github_url="https://github.com/Friend09",
    personal_writeup="""
    As an AI/ML Engineer with over 10 years of experience in Agile Project Management, 4 years in data science, and 10 years in business systems analysis, I am excited to bring my extensive expertise to the Enterprise Data & Machine Learning (EDML) organization at CVS Health. My career has been defined by a commitment to delivering transformative data solutions, managing complex projects, and driving innovation in IT operations.

    Throughout my career, I have successfully identified customer needs and translated them into actionable product visions that drive value across various business lines. My ability to communicate technical concepts to non-technical stakeholders has been a cornerstone of my success, allowing me to bridge the gap between business and technology seamlessly. I excel in developing and managing product roadmaps, prioritizing feature enhancements, and steering high-performing product lines to achieve cross-functional operational excellence.

    My experience in cloud technologies, particularly AWS, coupled with my proficiency in Agile Scrum methodologies, enables me to navigate development teams through organizational challenges and deliver automation and intelligent recommendations that improve product outcomes. I have a proven track record of leading cross-functional initiatives from planning through to launch, ensuring successful execution and continuous improvement.

    At CVS Health, I am eager to leverage my technical product leadership to contribute to the development of innovative data products that align with the company’s mission of delivering human-centric health care. I am passionate about championing product management strategies across the enterprise, collaborating with stakeholders, and driving initiatives that enhance the efficiency and effectiveness of healthcare solutions.

    With a strong foundation in machine learning, AI-driven solutions, and MLOps practices, I have implemented various projects that have significantly improved IT operations and service management. My work at Marsh & McLennan Company, where I led AI-based projects and developed end-to-end solutions for ServiceNow support ticket analysis, showcases my ability to deliver impactful results.

    I am enthusiastic about the opportunity to join CVS Health and contribute to its vision of making healthcare more personal, convenient, and affordable. I am confident that my skills, experience, and passion for innovation will make a valuable addition to the EDML organization and the broader CVS Health team.""",
    output_resume_file=f"/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_AI/dev_proj_AI/pract-crewai/files/resume_{JOB_PROFILE}.md",
    output_interview_file=f"/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_AI/dev_proj_AI/pract-crewai/files/interview_{JOB_PROFILE}.md"
)
