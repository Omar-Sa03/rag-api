from fastapi import HTTPException, status

class RAGException(HTTPException):
    def __init__(self, detail: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(status_code=status_code, detail=detail)

class DocumentProcessingError(RAGException):
    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

class VectorDBError(RAGException):
    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

class LLMGenerationError(RAGException):
    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=status.HTTP_502_BAD_GATEWAY)

class InvalidSearchModeError(RAGException):
    def __init__(self, allowed_modes: list):
        super().__init__(
            detail=f"Invalid search mode. Allowed: {allowed_modes}",
            status_code=status.HTTP_400_BAD_REQUEST
        )
