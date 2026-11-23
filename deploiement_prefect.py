from prefect import serve
from prefect.client.schemas.schedules import CronSchedule
from pipeline_prefect import flow_all, flow_train, flow_evaluate

if __name__ == "__main__":
    dep_all = flow_all.to_deployment(
        name="ml-pipeline-all",
        schedules=[CronSchedule(cron="0 2 * * *", timezone="Africa/Tunis")],
        tags=["ml", "all"],
    )
    dep_train = flow_train.to_deployment(
        name="ml-pipeline-train",
        tags=["ml", "train"],
    )
    dep_eval = flow_evaluate.to_deployment(
        name="ml-pipeline-evaluate",
        tags=["ml", "evaluate"],
    )
    serve(dep_all, dep_train, dep_eval)
