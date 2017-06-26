package utilis;

import java.io.Serializable;

/**
 * 
 * Detects operartional system . Retrieved from here : https://gist.github.com/kiuz/816e24aa787c2d102dd0
 *
 */
public class detectos implements Serializable{

    /**
	 * 
	 */
	private static final long serialVersionUID = 7673449933038592997L;
	
	private static String OS = System.getProperty("os.name").toLowerCase();

    public static boolean isWindows() {
        return (OS.indexOf("win") >= 0);
    }

    public static boolean isMac() {
        return (OS.indexOf("mac") >= 0);
    }

    public static boolean isUnix() {
        return (OS.indexOf("nix") >= 0 || OS.indexOf("nux") >= 0 || OS.indexOf("aix") > 0 );
    }

    public static boolean isSolaris() {
        return (OS.indexOf("sunos") >= 0);
    }
    public static String getOS(){
        if (isWindows()) {
            return "win";
        } else if (isMac()) {
            return "mac";
        } else if (isUnix()) {
            return "linux";
        } else if (isSolaris()) {
            return "sol";
        } else {
            return "err";
        }
    }

}