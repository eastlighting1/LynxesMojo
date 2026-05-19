/// Opaque internal node identifier for integer-addressed kernels.
///
/// This is not the same as the user-visible `_id` column, which is always `Utf8`.
pub type NodeId = u64;

/// Opaque internal edge identifier for integer-addressed kernels.
pub type EdgeId = u64;

pub const COL_NODE_ID: &str = "_id";
pub const COL_NODE_LABEL: &str = "_label";
pub const COL_EDGE_SRC: &str = "_src";
pub const COL_EDGE_DST: &str = "_dst";
pub const COL_EDGE_TYPE: &str = "_type";
pub const COL_EDGE_DIRECTION: &str = "_direction";

pub const NODE_RESERVED_COLUMNS: [&str; 2] = [COL_NODE_ID, COL_NODE_LABEL];
pub const EDGE_RESERVED_COLUMNS: [&str; 4] = [
    COL_EDGE_SRC,
    COL_EDGE_DST,
    COL_EDGE_TYPE,
    COL_EDGE_DIRECTION,
];

/// Canonical edge direction encoding used by `_direction`.
#[repr(i8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Out = 0,
    In = 1,
    Both = 2,
    None = 3,
}

impl Direction {
    pub const fn as_i8(self) -> i8 {
        self as i8
    }
}

impl TryFrom<i8> for Direction {
    type Error = crate::GFError;

    fn try_from(value: i8) -> crate::Result<Self> {
        match value {
            0 => Ok(Self::Out),
            1 => Ok(Self::In),
            2 => Ok(Self::Both),
            3 => Ok(Self::None),
            _ => Err(crate::GFError::InvalidDirection { value }),
        }
    }
}

impl From<Direction> for i8 {
    fn from(value: Direction) -> Self {
        value.as_i8()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_round_trips_to_i8() {
        assert_eq!(i8::from(Direction::Out), 0);
        assert_eq!(i8::from(Direction::In), 1);
        assert_eq!(i8::from(Direction::Both), 2);
        assert_eq!(i8::from(Direction::None), 3);

        assert_eq!(Direction::try_from(0).unwrap(), Direction::Out);
        assert_eq!(Direction::try_from(1).unwrap(), Direction::In);
        assert_eq!(Direction::try_from(2).unwrap(), Direction::Both);
        assert_eq!(Direction::try_from(3).unwrap(), Direction::None);
    }

    #[test]
    fn invalid_direction_is_rejected() {
        let err = Direction::try_from(9).unwrap_err();
        assert!(matches!(err, crate::GFError::InvalidDirection { value: 9 }));
    }

    #[test]
    fn reserved_columns_are_ordered_canonically() {
        assert_eq!(NODE_RESERVED_COLUMNS, ["_id", "_label"]);
        assert_eq!(
            EDGE_RESERVED_COLUMNS,
            ["_src", "_dst", "_type", "_direction"]
        );
    }
}
