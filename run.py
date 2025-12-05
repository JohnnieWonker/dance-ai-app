import os
import sys


os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
os.environ["STREAMLIT_SERVER_PORT"] = "8501"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"


def resource_path(relative_path: str) -> str:
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def main():
    app_path = resource_path(os.path.join("dance_ai_app", "pro_app.py"))
    os.chdir(os.path.dirname(app_path))
    from streamlit.web import cli as stcli
    sys.argv = [
        "streamlit",
        "run",
        app_path,
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
