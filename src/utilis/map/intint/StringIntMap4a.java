package utilis.map.intint;

/**
 * based as IntIntMap4, to work as a String to int map . This is no 100% in all cases , 
 * but it is used temporalilly for some parts of the project 
 */
public class StringIntMap4a  implements StringIntMap
{
    private static final int FREE_KEY = "".hashCode();

    public static final int NO_VALUE = 0;

    /** Keys and values */
    private int[] m_data;

    /** Do we have 'free' key in the map? */
    private boolean m_hasFreeKey;
    /** Value of 'free' key */
    private int m_freeValue;

    /** Fill factor, must be between (0 and 1) */
    private final float m_fillFactor;
    /** We will resize a map once it reaches this size */
    private int m_threshold;
    /** Current map size */
    private int m_size;

    /** Mask to calculate the original position */
    private int m_mask;
    private int m_mask2;

    public StringIntMap4a( final int size, final float fillFactor )
    {
        if ( fillFactor <= 0 || fillFactor >= 1 )
            throw new IllegalArgumentException( "FillFactor must be in (0, 1)" );
        if ( size <= 0 )
            throw new IllegalArgumentException( "Size must be positive!" );
        final int capacity = Tools.arraySize(size, fillFactor);
        m_mask = capacity - 1;
        m_mask2 = capacity*2 - 1;
        m_fillFactor = fillFactor;

        m_data = new int[capacity * 2];
        m_threshold = (int) (capacity * fillFactor);
    }

    public int get( final String key )
    {
        int ptr = ( Tools.phiMix( key.hashCode() ) & m_mask) << 1;

        if ( key.hashCode() == FREE_KEY )
            return m_hasFreeKey ? m_freeValue : NO_VALUE;

        int k = m_data[ ptr ];

        if ( k == FREE_KEY )
            return NO_VALUE;  //end of chain already
        if ( k == key.hashCode() ) //we check FREE prior to this call
            return m_data[ ptr + 1 ];

        while ( true )
        {
            ptr = (ptr + 2) & m_mask2; //that's next index
            k = m_data[ ptr ];
            if ( k == FREE_KEY )
                return NO_VALUE;
            if ( k == key.hashCode() )
                return m_data[ ptr + 1 ];
        }
    }

    public int put( final String key, final int value )
    {
        if ( key.hashCode() == FREE_KEY )
        {
            final int ret = m_freeValue;
            if ( !m_hasFreeKey )
                ++m_size;
            m_hasFreeKey = true;
            m_freeValue = value;
            return ret;
        }

        int ptr = ( Tools.phiMix( key.hashCode() ) & m_mask) << 1;
        int k = m_data[ptr];
        if ( k == FREE_KEY ) //end of chain already
        {
            m_data[ ptr ] = key.hashCode();
            m_data[ ptr + 1 ] = value;
            if ( m_size >= m_threshold )
                rehash( m_data.length * 2 ); //size is set inside
            else
                ++m_size;
            return NO_VALUE;
        }
        else if ( k == key.hashCode() ) //we check FREE prior to this call
        {
            final int ret = m_data[ ptr + 1 ];
            m_data[ ptr + 1 ] = value;
            return ret;
        }

        while ( true )
        {
            ptr = ( ptr + 2 ) & m_mask2; //that's next index calculation
            k = m_data[ ptr ];
            if ( k == FREE_KEY )
            {
                m_data[ ptr ] = key.hashCode();
                m_data[ ptr + 1 ] = value;
                if ( m_size >= m_threshold )
                    rehash( m_data.length * 2 ); //size is set inside
                else
                    ++m_size;
                return NO_VALUE;
            }
            else if ( k == key.hashCode() )
            {
                final int ret = m_data[ ptr + 1 ];
                m_data[ ptr + 1 ] = value;
                return ret;
            }
        }
    }

    public int remove( final String key )
    {
        if ( key.hashCode() == FREE_KEY )
        {
            if ( !m_hasFreeKey )
                return NO_VALUE;
            m_hasFreeKey = false;
            --m_size;
            return m_freeValue; //value is not cleaned
        }

        int ptr = ( Tools.phiMix( key.hashCode() ) & m_mask) << 1;
        int k = m_data[ ptr ];
        if ( k == key.hashCode() ) //we check FREE prior to this call
        {
            final int res = m_data[ ptr + 1 ];
            shiftKeys( ptr );
            --m_size;
            return res;
        }
        else if ( k == FREE_KEY )
            return NO_VALUE;  //end of chain already
        while ( true )
        {
            ptr = ( ptr + 2 ) & m_mask2; //that's next index calculation
            k = m_data[ ptr ];
            if ( k == key.hashCode() )
            {
                final int res = m_data[ ptr + 1 ];
                shiftKeys( ptr );
                --m_size;
                return res;
            }
            else if ( k == FREE_KEY )
                return NO_VALUE;
        }
    }

    private int shiftKeys(int pos)
    {
        // Shift entries with the same hash.
        int last, slot;
        int k;
        final int[] data = this.m_data;
        while ( true )
        {
            pos = ((last = pos) + 2) & m_mask2;
            while ( true )
            {
                if ((k = data[pos]) == FREE_KEY)
                {
                    data[last] = FREE_KEY;
                    return last;
                }
                slot = ( Tools.phiMix( k ) & m_mask) << 1; //calculate the starting slot for the current key
                if (last <= pos ? last >= slot || slot > pos : last >= slot && slot > pos) break;
                pos = (pos + 2) & m_mask2; //go to the next entry
            }
            data[last] = k;
            data[last + 1] = data[pos + 1];
        }
    }


    public int size()
    {
        return m_size;
    }

    private void rehash( final int newCapacity )
    {
        m_threshold = (int) (newCapacity/2 * m_fillFactor);
        m_mask = newCapacity/2 - 1;
        m_mask2 = newCapacity - 1;

        final int oldCapacity = m_data.length;
        final int[] oldData = m_data;

        m_data = new int[ newCapacity ];
        m_size = m_hasFreeKey ? 1 : 0;

        for ( int i = 0; i < oldCapacity; i += 2 ) {
            final int oldKey = oldData[ i ];
            if( oldKey != FREE_KEY )
                put( oldKey + "", oldData[ i + 1 ]);
        }
    }



//    private int getStartIdx( final int key )
//    {
//        return ( Tools.phiMix( key ) & m_mask) << 1;
//    }
}

