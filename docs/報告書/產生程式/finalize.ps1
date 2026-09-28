param([string]$InDocx, [string]$OutDocx, [string]$OutPdf)
# 用 Word 更新目錄與頁碼，另存最終 docx 與 PDF，回報總頁數
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($InDocx, $false, $false)
    foreach ($t in $doc.TablesOfContents) { $t.Update() }
    $doc.Fields.Update() | Out-Null
    foreach ($t in $doc.TablesOfContents) { $t.Update() }   # 頁碼可能因目錄長度改變，再更新一次
    # 競賽匿名規定：移除作者、最後儲存者等個人資訊，並讓之後每次儲存都自動移除
    $doc.RemoveDocumentInformation(99)  # wdRDIAll
    $doc.RemovePersonalInformation = $true
    $doc.SaveAs2($OutDocx, 16)          # wdFormatXMLDocument
    $doc.ExportAsFixedFormat($OutPdf, 17)  # wdExportFormatPDF
    $pages = $doc.ComputeStatistics(2)     # wdStatisticPages
    "總頁數：$pages"
    $doc.Close($false)
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
