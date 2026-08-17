from src.core.db import get_sql_database


class BankingRepository:

    """
    Repository for banking database operations.
    """

    def get_database(self):
        return get_sql_database()


    def get_customer_accounts(self):
        db = self.get_database()

        query = """
        SELECT *
        FROM accounts
        """

        return db.run(query)


    def execute_query(
        self,
        query: str
    ):

        db = self.get_database()

        return db.run(query)