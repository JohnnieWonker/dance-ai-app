import os
import sys


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

    # 切换到应用所在目录，避免相对路径问题
    os.chdir(os.path.dirname(app_path))

    # 重要：在当前进程里调用 streamlit 的 CLI 入口
    from streamlit.web import cli as stcli

    # 模拟命令行参数：streamlit run dance_ai_app/pro_app.py --server.port=8501
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port=8501",
    ]

    # 运行 streamlit（会自己打开浏览器）
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
