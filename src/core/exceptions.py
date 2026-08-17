class SmartBankingException(Exception):
    """Base exception for the Smart Banking Assistant."""


class ConfigurationError(SmartBankingException):
    """Raised when application configuration is invalid."""


class DatabaseError(SmartBankingException):
    """Raised when a database operation fails."""


class IngestionError(SmartBankingException):
    """Raised when document ingestion fails."""


class RetrievalError(SmartBankingException):
    """Raised when retrieval fails."""


class SQLValidationError(SmartBankingException):
    """Raised when generated SQL fails validation."""


class AgentError(SmartBankingException):
    """Raised when an agent operation fails."""