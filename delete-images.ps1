# Helper script to delete specific training images
# Usage:
#   1. List all images: .\delete-images.ps1 list
#   2. Delete specific: .\delete-images.ps1 delete 001 002 015 023
#   3. Keep only: .\delete-images.ps1 keep-only 001 003 005 010

param(
    [Parameter(Mandatory=$true)]
    [string]$Action,

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Numbers
)

$imagesDir = "C:\projects\oykh-temp\lora-training-v2\images"
$captionsDir = "C:\projects\oykh-temp\lora-training-v2\captions"

switch ($Action.ToLower()) {
    "list" {
        Write-Host "`nTraining Images:" -ForegroundColor Cyan
        Write-Host "================`n" -ForegroundColor Cyan

        Get-ChildItem $imagesDir -Filter *.png | Sort-Object Name | ForEach-Object {
            $num = $_.Name.Substring(6, 3)  # Extract number from train_XXX_
            $size = [math]::Round($_.Length / 1MB, 2)
            Write-Host "$num - $($_.Name) ($size MB)"
        }

        $count = (Get-ChildItem $imagesDir -Filter *.png).Count
        Write-Host "`nTotal: $count images`n" -ForegroundColor Green
    }

    "delete" {
        if ($Numbers.Count -eq 0) {
            Write-Host "Error: Specify image numbers to delete" -ForegroundColor Red
            Write-Host "Example: .\delete-images.ps1 delete 001 002 015"
            exit
        }

        foreach ($num in $Numbers) {
            $pattern = "train_$($num.PadLeft(3,'0'))_*.png"
            $files = Get-ChildItem $imagesDir -Filter $pattern

            foreach ($file in $files) {
                Remove-Item $file.FullName -Force
                Write-Host "Deleted: $($file.Name)" -ForegroundColor Yellow

                # Delete corresponding caption
                $captionFile = Join-Path $captionsDir ($file.BaseName + ".txt")
                if (Test-Path $captionFile) {
                    Remove-Item $captionFile -Force
                }
            }
        }

        $remaining = (Get-ChildItem $imagesDir -Filter *.png).Count
        Write-Host "`nRemaining: $remaining images`n" -ForegroundColor Green
    }

    "keep-only" {
        if ($Numbers.Count -eq 0) {
            Write-Host "Error: Specify image numbers to KEEP" -ForegroundColor Red
            Write-Host "Example: .\delete-images.ps1 keep-only 001 005 010 015 020"
            exit
        }

        Write-Host "WARNING: This will delete all images EXCEPT: $($Numbers -join ', ')" -ForegroundColor Yellow
        $confirm = Read-Host "Continue? (yes/no)"

        if ($confirm -ne "yes") {
            Write-Host "Cancelled" -ForegroundColor Red
            exit
        }

        $keepPatterns = $Numbers | ForEach-Object { "train_$($_.PadLeft(3,'0'))_" }
        $allFiles = Get-ChildItem $imagesDir -Filter *.png

        foreach ($file in $allFiles) {
            $shouldKeep = $false
            foreach ($pattern in $keepPatterns) {
                if ($file.Name.StartsWith($pattern)) {
                    $shouldKeep = $true
                    break
                }
            }

            if (-not $shouldKeep) {
                Remove-Item $file.FullName -Force
                Write-Host "Deleted: $($file.Name)" -ForegroundColor Yellow

                # Delete corresponding caption
                $captionFile = Join-Path $captionsDir ($file.BaseName + ".txt")
                if (Test-Path $captionFile) {
                    Remove-Item $captionFile -Force
                }
            }
        }

        $remaining = (Get-ChildItem $imagesDir -Filter *.png).Count
        Write-Host "`nKept: $remaining images`n" -ForegroundColor Green
    }

    default {
        Write-Host "Unknown action: $Action" -ForegroundColor Red
        Write-Host "`nUsage:" -ForegroundColor Cyan
        Write-Host "  .\delete-images.ps1 list"
        Write-Host "  .\delete-images.ps1 delete 001 002 015 023"
        Write-Host "  .\delete-images.ps1 keep-only 001 003 005 010 015"
    }
}
