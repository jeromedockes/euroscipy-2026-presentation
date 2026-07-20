import skrub

# %%
skrub.tabular_pipeline("regressor")


# %%
df = skrub.datasets.fetch_employee_salaries()['employee_salaries'].sample(frac=1, random_state=0)
skrub.set_config(table_report_n_rows=5, table_report_verbosity=0)

# %%
skrub.TableReport(df)
