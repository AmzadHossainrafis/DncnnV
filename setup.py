import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


def get_requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()  # ["req1==1.0.0", "req2==0.0.8"]


__version__ = "1.0.0"

REPO_NAME = "CAMVID"
AUTHOR_USER_NAME = "Amzad hossain"
SRC_REPO = "Dncnn"
AUTHOR_EMAIL = "amzad.rafi@northsouth.edu"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for image de-blurring using DnCNN",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
