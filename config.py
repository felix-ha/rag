from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    resources_path: str = Field(alias='PATH_RESOURCES')
    state_file: str = Field(alias='STATE_FILE')
