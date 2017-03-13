package utilis.map.intint;

/**
 * These methods will be implemented by all test maps
 */
public interface StringIntMap {
    public int get( final String key );
    public int put( final String key, final int value );
    public int remove( final String key );
    public int size();
}
