from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


class PromptLog(db.Model):
    __tablename__ = 'prompt_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    model = db.Column(db.String(100), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    prompt_tokens = db.Column(db.Integer)
    response_tokens = db.Column(db.Integer)
    total_tokens = db.Column(db.Integer)
    
    def to_dict(self):
        """Convert the object to a dictionary for API responses"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'model': self.model,
            'prompt': self.prompt,
            'response': self.response,
            'estimated_tokens': {
                'prompt': self.prompt_tokens,
                'response': self.response_tokens,
                'total': self.total_tokens
            }
        }