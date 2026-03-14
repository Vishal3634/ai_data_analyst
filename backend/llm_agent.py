import os
import re
import json
import base64
import logging
import sqlite3
import traceback
from io import BytesIO

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CHART_KEYWORDS = [
    "plot", "chart", "graph", "visualize", "visualise",
    "show", "display", "draw", "distribution", "trend",
    "bar", "line", "histogram", "pie", "scatter"
]

SQL_KEYWORDS = [
    "sql", "query", "select", "where", "group by",
    "join", "sum", "count", "max", "min", "average",
    "highest", "lowest", "top", "bottom", "total",
    "which city", "which product", "which region"
]


class DataAnalystAgent:

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found! Please add it to your .env file")

        # ✅ Faster model + reduced tokens for speed
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama3-8b-8192",
            temperature=0,
            max_tokens=1024
        )
        logger.info("✅ DataAnalystAgent ready with Groq")

    def run(self, question: str, datasets: dict) -> dict:
        question_lower = question.lower()
        is_chart = any(kw in question_lower for kw in CHART_KEYWORDS)
        is_sql   = any(kw in question_lower for kw in SQL_KEYWORDS)

        if is_chart:
            return self._handle_chart_query(question, datasets)
        elif is_sql:
            return self._handle_sql_query(question, datasets)
        else:
            return self._handle_pandas_query(question, datasets)

    def _handle_pandas_query(self, question: str, datasets: dict) -> dict:
        try:
            df = list(datasets.values())[0]

            agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df=df,
                verbose=True,
                agent_type="zero-shot-react-description",
                allow_dangerous_code=True,
                max_iterations=8,               # ✅ balanced
                max_execution_time=30,          # ✅ 30s timeout
                early_stopping_method="generate" # ✅ clean stop
            )

            result = agent.invoke({"input": question})
            answer = result.get("output", str(result))

            return {
                "answer": answer,
                "chart": None,
                "query_type": "pandas",
                "sql_query": None
            }

        except Exception as e:
            logger.error(f"Pandas agent error: {traceback.format_exc()}")
            return {
                "answer": f"Error analyzing data: {str(e)}",
                "chart": None,
                "query_type": "pandas",
                "sql_query": None
            }

    def _handle_sql_query(self, question: str, datasets: dict) -> dict:
        db_path = "/tmp/temp_analysis.db"
        sql_query_used = None

        try:
            conn = sqlite3.connect(db_path)

            for filename, df in datasets.items():
                table_name = (filename
                              .replace(".csv", "")
                              .replace("-", "_")
                              .replace(" ", "_"))
                df.to_sql(table_name, conn, if_exists="replace", index=False)

            conn.close()

            db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
            sql_agent = create_sql_agent(
                llm=self.llm,
                db=db,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=8,               # ✅ balanced
                max_execution_time=30,          # ✅ 30s timeout
                early_stopping_method="generate" # ✅ clean stop
            )

            result = sql_agent.invoke({"input": question})
            answer = result.get("output", str(result))

            return {
                "answer": answer,
                "chart": None,
                "query_type": "sql",
                "sql_query": sql_query_used
            }

        except Exception as e:
            logger.error(f"SQL agent error: {traceback.format_exc()}")
            result = self._handle_pandas_query(question, datasets)
            result["query_type"] = "pandas_fallback"
            return result

    def _handle_chart_query(self, question: str, datasets: dict) -> dict:
        df = list(datasets.values())[0]

        try:
            column_prompt = f"""
You are a data analyst. Given these DataFrame columns: {list(df.columns)}
And this user request: "{question}"

Respond ONLY with a valid JSON object (no markdown, no explanation):
{{
  "chart_type": "bar",
  "x_column": "column_name",
  "y_column": "column_name",
  "title": "chart title"
}}

Rules:
- chart_type must be one of: bar, line, histogram, scatter, pie
- x_column and y_column must EXACTLY match one of the column names above
- For histogram set y_column to null
"""
            response = self.llm.invoke(column_prompt)
            raw = response.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            chart_config = json.loads(raw)

            chart_type = chart_config.get("chart_type", "bar")
            x_col      = chart_config.get("x_column")
            y_col      = chart_config.get("y_column")
            title      = chart_config.get("title", question)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.set_style("whitegrid")

            if chart_type == "bar" and x_col and y_col:
                if df[x_col].dtype == "object":
                    plot_df = df.groupby(x_col)[y_col].sum().reset_index()
                    sns.barplot(data=plot_df, x=x_col, y=y_col, ax=ax, palette="viridis")
                else:
                    sns.barplot(data=df, x=x_col, y=y_col, ax=ax, palette="viridis")
                plt.xticks(rotation=45, ha="right")

            elif chart_type == "line" and x_col and y_col:
                plot_df = df.sort_values(x_col)
                sns.lineplot(data=plot_df, x=x_col, y=y_col, ax=ax, marker="o", color="#2ecc71")

            elif chart_type == "histogram" and x_col:
                sns.histplot(data=df, x=x_col, ax=ax, kde=True, color="#3498db")

            elif chart_type == "scatter" and x_col and y_col:
                sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, color="#e74c3c", alpha=0.7)

            elif chart_type == "pie" and x_col and y_col:
                pie_df = df.groupby(x_col)[y_col].sum()
                ax.pie(pie_df.values, labels=pie_df.index, autopct="%1.1f%%",
                       colors=sns.color_palette("viridis", len(pie_df)))
            else:
                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) > 0:
                    df[numeric_cols[0]].hist(ax=ax, color="#3498db", bins=20)
                    ax.set_xlabel(numeric_cols[0])

            ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close(fig)

            return {
                "answer": f"Chart generated: {title}",
                "chart": chart_base64,
                "query_type": "chart",
                "sql_query": None
            }

        except Exception as e:
            logger.error(f"Chart error: {traceback.format_exc()}")
            result = self._handle_pandas_query(question, datasets)
            result["query_type"] = "chart_fallback"
            return result

    def generate_insights(self, datasets: dict) -> list:
        insights = []

        for filename, df in datasets.items():
            try:
                summary = self._build_summary(df)
                prompt = f"""
You are a senior business data analyst. Analyze this dataset and provide insights.

Dataset: {filename}
{summary}

Provide exactly 5 actionable business insights.
Each insight must start with "•" and be 1-2 sentences.
Return ONLY the 5 bullet points, nothing else.
"""
                response = self.llm.invoke(prompt)
                raw_insights = response.content.strip()

                lines = [line.strip() for line in raw_insights.split("\n") if line.strip()]
                for line in lines:
                    if line.startswith(("•", "-", "*")):
                        insights.append(f"[{filename}] {line}")

            except Exception as e:
                logger.error(f"Insight error: {str(e)}")
                insights.append(f"[{filename}] Could not generate insights: {str(e)}")

        return insights if insights else ["No insights generated. Please check your data."]

    def _build_summary(self, df: pd.DataFrame) -> str:
        parts = []
        parts.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        parts.append(f"Columns: {list(df.columns)}")

        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            stats = numeric_df.describe().round(2)
            parts.append(f"Numeric Statistics:\n{stats.to_string()}")

        cat_cols = df.select_dtypes(include="object").columns
        for col in list(cat_cols)[:3]:
            top_vals = df[col].value_counts().head(5)
            parts.append(f"Top values in '{col}': {top_vals.to_dict()}")

        missing = df.isnull().sum()
        if missing.any():
            parts.append(f"Missing values: {missing[missing > 0].to_dict()}")

        return "\n".join(parts)