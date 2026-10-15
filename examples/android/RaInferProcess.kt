// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
package example.rainfer

import org.json.JSONArray
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.util.concurrent.TimeUnit

/** Blocking process adapter, not an application. Call start/generate/close off the UI thread.
 * One request at a time. The owner must impose deadlines and call cancel on timeout.
 */
class RaInferProcess private constructor(private val process: Process) : Closeable {
    private val input = process.outputStream.bufferedWriter(Charsets.UTF_8)
    private val output = process.inputStream.bufferedReader(Charsets.UTF_8)
    private var nextId = 1L
    val ready: JSONObject = readRecord().also {
        check(it.optString("event") == "ready") { "RaInfer startup failed: $it" }
    }

    private fun readRecord(): JSONObject {
        while (true) {
            val line = output.readLine() ?: error("RaInfer exited before a protocol response")
            if (line.startsWith("@DFLASH2 ")) return JSONObject(line.removePrefix("@DFLASH2 "))
        }
    }

    fun generate(tokens: LongArray, maxNewTokens: Int = 96, ngram: Boolean = true): JSONObject {
        require(tokens.isNotEmpty() && tokens.size <= 2044 && tokens.all { it in 0L until 262144L })
        val ids = JSONArray().apply { tokens.forEach { put(it) } }
        return request(JSONObject().put("tokens", ids), maxNewTokens, ngram)
    }

    /** Experimental image path; requires a vision-qualified PTE and a locally encoded bundle. */
    fun generateVision(bundle: File, maxNewTokens: Int = 96, ngram: Boolean = true): JSONObject {
        require(bundle.isFile) { "Missing encoded image bundle" }
        check(ready.optInt("vision_bundle_protocol", 0) == 1) { "Worker model is text-only" }
        return request(JSONObject().put("vision_bundle", bundle.absolutePath), maxNewTokens, ngram)
    }

    @Synchronized
    private fun request(request: JSONObject, maxNewTokens: Int, ngram: Boolean): JSONObject {
        check(process.isAlive) { "RaInfer worker is not running" }
        require(maxNewTokens in 1..2048)
        val id = nextId++
        request.put("id", id)
            .put("max_new_tokens", maxNewTokens).put("cpu_workers", 4).put("body_workers", 1)
            .put("fast_argmax", true).put("compact_logits", true).put("ngram", ngram)
            .put("gpu_head", false).put("burst", false).put("fixed_affinity", false)
        input.write(request.toString())
        input.newLine()
        input.flush()
        val result = readRecord()
        check(result.optLong("id", -1L) == id) { "Unexpected RaInfer response ID" }
        check(result.optString("event") == "result") { result.optString("message", "RaInfer request failed") }
        return result.getJSONObject("report")
    }

    /** May be called from another thread while generate waits for a response. */
    fun cancel() {
        process.destroyForcibly()
    }

    /** Graceful close when idle. Use cancel(), not close(), for an active generation. */
    @Synchronized
    override fun close() {
        try {
            if (process.isAlive) {
                input.write(JSONObject().put("id", nextId++).put("command", "shutdown").toString())
                input.newLine()
                input.flush()
                if (!process.waitFor(2, TimeUnit.SECONDS)) process.destroyForcibly()
            }
        } finally {
            if (process.isAlive) process.destroyForcibly()
            runCatching { input.close() }
            runCatching { output.close() }
        }
    }

    companion object {
        /** nativeLibraryDir is the APK install directory, not writable app data. */
        @JvmStatic
        fun start(
            nativeLibraryDir: File,
            runtimeDirectory: File,
            modelDirectory: File,
            workDirectory: File,
            targetFile: File = File(modelDirectory, "gemma4-e2b-dflash2-verify-2k.pte"),
            contractFile: File = File(modelDirectory, "io-contract.json"),
            diagnostics: (String) -> Unit = {}
        ): RaInferProcess {
            val executable = File(nativeLibraryDir, "libRaInferRunner.so")
            require(executable.isFile && executable.canExecute()) { "APK-packaged RaInfer executable missing" }
            require(runtimeDirectory.isDirectory && modelDirectory.isDirectory)
            require(workDirectory.isDirectory || workDirectory.mkdirs())
            val assets = listOf(targetFile, File(modelDirectory, "cpu-body-fp32.bin"),
                File(modelDirectory, "draft-manifest.json"), contractFile)
                .onEach { require(it.isFile) { "Missing ${it.name}" } }
            val builder = ProcessBuilder(listOf(executable.absolutePath, "--serve") + assets.map { it.absolutePath })
                .directory(workDirectory)
            builder.environment()["LD_LIBRARY_PATH"] = "${nativeLibraryDir.absolutePath}:${runtimeDirectory.absolutePath}:/vendor/lib64"
            builder.environment()["ADSP_LIBRARY_PATH"] = "${runtimeDirectory.absolutePath};/vendor/lib/rfsa/adsp;/vendor/dsp"
            val process = builder.start()
            // Drain stderr independently to avoid a full diagnostic pipe blocking inference.
            Thread({
                runCatching { process.errorStream.bufferedReader(Charsets.UTF_8).useLines { lines -> lines.forEach(diagnostics) } }
            }, "RaInfer-stderr").apply { isDaemon = true; start() }
            return try {
                RaInferProcess(process)
            } catch (error: Throwable) {
                process.destroyForcibly()
                throw error
            }
        }
    }
}
