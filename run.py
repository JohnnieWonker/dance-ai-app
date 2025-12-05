import os
import sys


# 关键：关闭开发模式 & 固定端口 & 关掉统计弹窗
os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
os.environ["STREAMLIT_SERVER_PORT"] = "8501"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"


def resource_path(relative_path: str) -> str:
    """
    兼容 PyInstaller 的资源路径：
    - 开发环境：返回当前文件所在目录 + relative_path
    - 打包后：返回 _MEIPASS 临时目录 + relative_path
    """
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def main():
    # 找到打包后的 app 路径
    app_path = resource_path(os.path.join("dance_ai_app", "pro_app.py"))

    # 切到应用目录，保证相对路径正常
    os.chdir(os.path.dirname(app_path))

    # 在当前进程里跑 Streamlit
    from streamlit.web import cli as stcli

    # 等价于命令：streamlit run pro_app.py
    sys.argv = [
        "streamlit",
        "run",
        app_path,
    ]

    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
