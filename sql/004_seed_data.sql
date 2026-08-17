INSERT INTO customers
(
    customer_number,
    first_name,
    last_name,
    email
)
VALUES
(
    'CUST10001',
    'Rahul',
    'Sharma',
    'rahul@test.com'
)
ON CONFLICT DO NOTHING;



INSERT INTO accounts
(
    customer_id,
    account_number,
    account_type,
    balance
)
SELECT
    id,
    'ACC100001',
    'SAVINGS',
    50000
FROM customers
WHERE customer_number='CUST10001'
ON CONFLICT DO NOTHING;



INSERT INTO transactions
(
    account_id,
    transaction_reference,
    transaction_type,
    amount,
    description
)
SELECT
    id,
    'TXN10001',
    'DEBIT',
    1000,
    'ATM Withdrawal'
FROM accounts
WHERE account_number='ACC100001';