-- Create sample financial database structure
CREATE TABLE IF NOT EXISTS company_financials (
    id INT PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(10),
    company_name VARCHAR(100),
    quarter VARCHAR(10),
    year INT,
    revenue DECIMAL(15,2),
    net_income DECIMAL(15,2),
    total_assets DECIMAL(15,2),
    total_equity DECIMAL(15,2),
    debt_to_equity DECIMAL(5,2),
    roe DECIMAL(5,2),
    created_date DATE
);

-- Insert sample data
INSERT INTO company_financials VALUES
(1, 'RELIANCE.NS', 'Reliance Industries', 'Q1', 2024, 230000, 18000, 950000, 580000, 0.45, 15.2, '2024-04-01'),
(2, 'TCS.NS', 'Tata Consultancy Services', 'Q1', 2024, 59000, 11500, 180000, 145000, 0.05, 41.5, '2024-04-01'),
(3, 'INFY.NS', 'Infosys Limited', 'Q1', 2024, 38500, 7200, 95000, 78000, 0.02, 28.8, '2024-04-01');

INSERT INTO company_financials VALUES
(4, 'RELIANCE.NS', 'Reliance Industries', 'Q4', 2023, 220000, 17000, 940000, 570000, 0.46, 14.5, '2023-12-31'),
(5, 'TCS.NS',      'Tata Consultancy Services', 'Q4', 2023, 56000, 11000, 179000, 144000, 0.05, 40.0, '2023-12-31'),
(6, 'INFY.NS',     'Infosys Limited',            'Q4', 2023, 37000, 7000, 94000, 77000, 0.02, 27.5, '2023-12-31');


-- Query 1: Top performing companies by ROE
SELECT 
    company_name,
    ticker,
    roe,
    net_income,
    RANK() OVER (ORDER BY roe DESC) as roe_rank
FROM company_financials 
WHERE year = 2024 AND quarter = 'Q1'
ORDER BY roe DESC;

WITH revenue_growth AS (
    SELECT 
        ticker,
        company_name,
        year,
        quarter,
        revenue,
        LAG(revenue) OVER (PARTITION BY ticker ORDER BY year, quarter) AS prev_revenue,
        CASE 
            WHEN LAG(revenue) OVER (PARTITION BY ticker ORDER BY year, quarter) IS NULL THEN NULL
            ELSE ((revenue - LAG(revenue) OVER (PARTITION BY ticker ORDER BY year, quarter)) * 100.0
                    / NULLIF(LAG(revenue) OVER (PARTITION BY ticker ORDER BY year, quarter), 0))
        END AS growth_rate
    FROM company_financials
)
SELECT 
    ticker,
    company_name,
    revenue,
    prev_revenue,
    ROUND(growth_rate, 2) AS revenue_growth_percent
FROM revenue_growth 
WHERE growth_rate IS NOT NULL
ORDER BY growth_rate DESC;


-- Query 3: Financial health analysis
SELECT 
    ticker,
    company_name,
    revenue / 1000 as revenue_k_cr,
    net_income / 1000 as profit_k_cr,
    ROUND((net_income * 100.0) / revenue, 2) AS profit_margin,
    debt_to_equity,
    roe,
    CASE 
        WHEN roe > 20 AND debt_to_equity < 0.5 THEN 'Strong'
        WHEN roe > 15 AND debt_to_equity < 1.0 THEN 'Good'
        WHEN roe > 10 THEN 'Average'
        ELSE 'Weak'
    END as financial_health
FROM company_financials 
WHERE year = 2024 AND quarter = 'Q1'
ORDER BY roe DESC;