param([int]$Port = 8000)

try {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::IPv6Any, $Port)
    $listener.Server.DualMode = $true
    $listener.Start()
} catch {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $Port)
    $listener.Start()
}


Write-Output "=================================================================="
Write-Output " [ONLINE] Maritime AI Server running at:"
Write-Output " -> http://localhost:$Port/"
Write-Output " -> http://127.0.0.1:$Port/"
Write-Output "=================================================================="

$mimeMap = @{
    ".html"  = "text/html; charset=utf-8"
    ".htm"   = "text/html; charset=utf-8"
    ".css"   = "text/css; charset=utf-8"
    ".js"    = "application/javascript; charset=utf-8"
    ".json"  = "application/json; charset=utf-8"
    ".png"   = "image/png"
    ".jpg"   = "image/jpeg"
    ".jpeg"  = "image/jpeg"
    ".svg"   = "image/svg+xml"
    ".ico"   = "image/x-icon"
    ".mp4"   = "video/mp4"
    ".woff"  = "font/woff"
    ".woff2" = "font/woff2"
    ".ttf"   = "font/ttf"
    ".pdf"   = "application/pdf"
}


$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }

$buf = New-Object byte[] 4096

try {
    while ($true) {
        $client = $listener.AcceptTcpClient()
        $client.ReceiveTimeout = 5000
        $client.SendTimeout = 5000
        $stream = $client.GetStream()

        try {
            $bytesRead = $stream.Read($buf, 0, $buf.Length)
            if ($bytesRead -gt 0) {
                $rawReq = [System.Text.Encoding]::ASCII.GetString($buf, 0, $bytesRead)
                $firstLine = ($rawReq -split "`r`n")[0]
                $parts = $firstLine -split " "

                if ($parts.Length -ge 2) {
                    $reqPath = $parts[1].Split("?")[0]
                    if ($reqPath -eq "/" -or $reqPath -eq "") {
                        $reqPath = "/index.html"
                    }

                    if ($reqPath -eq "/api/status") {
                        $jsonResp = '{"status":"online","model":"Statoil Custom CNN + Cyclone LSTM Trajectory Predictor","server":"Maritime-AI-PS"}'
                        $jsonBytes = [System.Text.Encoding]::UTF8.GetBytes($jsonResp)
                        $header = "HTTP/1.1 200 OK`r`nContent-Type: application/json; charset=utf-8`r`nContent-Length: $($jsonBytes.Length)`r`nAccess-Control-Allow-Origin: *`r`nConnection: close`r`n`r`n"
                        $headerBytes = [System.Text.Encoding]::ASCII.GetBytes($header)
                        $stream.Write($headerBytes, 0, $headerBytes.Length)
                        $stream.Write($jsonBytes, 0, $jsonBytes.Length)
                    } else {
                        $decoded = [System.Uri]::UnescapeDataString($reqPath)
                        $rel = $decoded.TrimStart("/\").Replace("/", [System.IO.Path]::DirectorySeparatorChar)
                        $filePath = [System.IO.Path]::Combine($root, $rel)

                        if (-not [System.IO.File]::Exists($filePath)) {
                            $cand1 = [System.IO.Path]::Combine($root, "web", $rel)
                            $cand2 = [System.IO.Path]::Combine($root, "statoil-iceberg-classifier-challenge", "web", $rel)
                            if ([System.IO.File]::Exists($cand1)) {
                                $filePath = $cand1
                            } elseif ([System.IO.File]::Exists($cand2)) {
                                $filePath = $cand2
                            }
                        }

                        if ([System.IO.File]::Exists($filePath)) {
                            $ext = [System.IO.Path]::GetExtension($filePath).ToLower()
                            $mime = if ($mimeMap.ContainsKey($ext)) { $mimeMap[$ext] } else { "application/octet-stream" }
                            $fileLen = (New-Object System.IO.FileInfo($filePath)).Length

                            $header = "HTTP/1.1 200 OK`r`nContent-Type: $mime`r`nContent-Length: $fileLen`r`nAccess-Control-Allow-Origin: *`r`nConnection: close`r`n`r`n"
                            $headerBytes = [System.Text.Encoding]::ASCII.GetBytes($header)
                            $stream.Write($headerBytes, 0, $headerBytes.Length)

                            $fs = [System.IO.File]::OpenRead($filePath)
                            try {
                                $chunk = New-Object byte[] 65536
                                while (($count = $fs.Read($chunk, 0, $chunk.Length)) -gt 0) {
                                    $stream.Write($chunk, 0, $count)
                                }
                            } finally {
                                $fs.Close()
                            }
                        } else {
                            $msg = "404 Not Found"
                            $msgBytes = [System.Text.Encoding]::UTF8.GetBytes($msg)
                            $header = "HTTP/1.1 404 Not Found`r`nContent-Type: text/plain`r`nContent-Length: $($msgBytes.Length)`r`nAccess-Control-Allow-Origin: *`r`nConnection: close`r`n`r`n"
                            $headerBytes = [System.Text.Encoding]::ASCII.GetBytes($header)
                            $stream.Write($headerBytes, 0, $headerBytes.Length)
                            $stream.Write($msgBytes, 0, $msgBytes.Length)
                        }
                    }
                    $stream.Flush()
                }
            }
        } catch {
        } finally {
            $client.Close()
        }
    }
} finally {
    $listener.Stop()
}
