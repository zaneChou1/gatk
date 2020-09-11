package org.broadinstitute.hellbender.utils.runtime;

/**
 * Python script execution exception that indicates that python script needs to be restarted.
 */
public class RestartScriptExecutorException extends ScriptExecutorException {

    private static final long serialVersionUID = 0L;

    public RestartScriptExecutorException(String msg) {
        super(msg);
    }
}
