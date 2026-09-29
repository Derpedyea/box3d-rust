# Pre-commit check script for Box3D Rust Port
# This script runs file length validation and other checks before commits

param(
    [switch]$Fix = $false
)

Write-Host "[*] Running pre-commit checks for Box3D Rust Port..." -ForegroundColor Cyan

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Check if cargo is available
if (!(Test-Command "cargo")) {
    Write-Host "[FAIL] Cargo is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

$exitCode = 0

Write-Host "`n[*] Running file length validation..." -ForegroundColor Yellow
try {
    cargo test --test file_length_validation --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] All files are within the 800-line limit" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Some files exceed the 800-line limit" -ForegroundColor Red
        Write-Host "   Run 'cargo test --test file_length_validation file_metrics::generate_refactoring_report' for refactoring suggestions" -ForegroundColor Yellow
        $exitCode = 1
    }
} catch {
    Write-Host "[FAIL] File length validation failed: $_" -ForegroundColor Red
    $exitCode = 1
}

Write-Host "`n[*] Running unit tests..." -ForegroundColor Yellow
try {
    cargo test --lib --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] All unit tests passed" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Some unit tests failed" -ForegroundColor Red
        $exitCode = 1
    }
} catch {
    Write-Host "[FAIL] Unit tests failed: $_" -ForegroundColor Red
    $exitCode = 1
}

# Skip integration tests if they don't exist or only contain file_length_validation
if (Test-Path "Tests") {
    $integrationTests = Get-ChildItem -Path "Tests" -Filter "*.rs" | Where-Object { $_.Name -ne "file_length_validation.rs" }
    if ($integrationTests.Count -gt 0) {
        Write-Host "`n[*] Running integration tests..." -ForegroundColor Yellow
        try {
            cargo test --test "*" --quiet
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[OK] Integration tests passed" -ForegroundColor Green
            } else {
                Write-Host "[FAIL] Integration tests failed" -ForegroundColor Red
                $exitCode = 1
            }
        } catch {
            Write-Host "[FAIL] Integration tests failed: $_" -ForegroundColor Red
            $exitCode = 1
        }
    }
}

if (Test-Command "cargo-fmt") {
    Write-Host "`n[*] Checking code formatting..." -ForegroundColor Yellow
    try {
        cargo fmt --all -- --check
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Code formatting is correct" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] Code formatting issues found" -ForegroundColor Red
            if ($Fix) {
                Write-Host "[*] Fixing code formatting..." -ForegroundColor Yellow
                cargo fmt --all
                Write-Host "[OK] Code formatting fixed" -ForegroundColor Green
            } else {
                Write-Host "   Run 'cargo fmt --all' to fix formatting or use -Fix flag" -ForegroundColor Yellow
                $exitCode = 1
            }
        }
    } catch {
        Write-Host "[FAIL] Code formatting check failed: $_" -ForegroundColor Red
        $exitCode = 1
    }
} else {
    Write-Host "[WARN]  rustfmt not available, skipping formatting check" -ForegroundColor Yellow
}

if (Test-Command "cargo-clippy") {
    Write-Host "`n[*] Running clippy lints..." -ForegroundColor Yellow
    try {
        cargo clippy --all-targets --all-features -- -D warnings
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] No clippy warnings found" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] Clippy warnings found" -ForegroundColor Red
            $exitCode = 1
        }
    } catch {
        Write-Host "[FAIL] Clippy check failed: $_" -ForegroundColor Red
        $exitCode = 1
    }
} else {
    Write-Host "[WARN]  clippy not available, skipping lint check" -ForegroundColor Yellow
}

Write-Host "`n[*]  Running build check..." -ForegroundColor Yellow
try {
    cargo build --all-targets
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Build successful" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Build failed" -ForegroundColor Red
        $exitCode = 1
    }
} catch {
    Write-Host "[FAIL] Build check failed: $_" -ForegroundColor Red
    $exitCode = 1
}

# Check for benchmark build if benchmarks exist
if (Test-Path "benches") {
    Write-Host "`n[*] Running benchmark build check..." -ForegroundColor Yellow
    try {
        cargo bench --no-run
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Benchmark build successful" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] Benchmark build failed" -ForegroundColor Red
            $exitCode = 1
        }
    } catch {
        Write-Host "[FAIL] Benchmark build check failed: $_" -ForegroundColor Red
        $exitCode = 1
    }
}

Write-Host "`n" -NoNewline
if ($exitCode -eq 0) {
    Write-Host "[OK] All pre-commit checks passed!" -ForegroundColor Green
    Write-Host "   Your Box3D Rust code is ready for commit." -ForegroundColor Green
} else {
    Write-Host "[FAIL] Pre-commit checks failed!" -ForegroundColor Red
    Write-Host "   Please fix the issues above before committing." -ForegroundColor Red
    Write-Host "`n[*] Helpful commands:" -ForegroundColor Cyan
    Write-Host "   - cargo test --verbose                    - Run tests with detailed output" -ForegroundColor White
    Write-Host "   - cargo fmt --all                         - Fix formatting issues" -ForegroundColor White
    Write-Host "   - cargo clippy --fix --all-targets        - Fix clippy warnings automatically" -ForegroundColor White
    Write-Host "   - cargo test file_length_validation       - Check file lengths" -ForegroundColor White
    Write-Host "   - cargo bench --no-run                    - Build benchmarks without running" -ForegroundColor White
}

exit $exitCode