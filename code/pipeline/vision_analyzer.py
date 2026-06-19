import json
import re
import google.generativeai as genai
from tenacity import retry, wait_exponential, stop_after_attempt
